import random
from collections import defaultdict
from math import prod
from typing import Callable, Dict, List, Protocol, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
from frozendict import frozendict
from tqdm import tqdm

from jaxip.datasets.base import StructureDataset
from jaxip.logger import logger
from jaxip.potentials._energy import _compute_energy
from jaxip.potentials._force import _compute_force
from jaxip.potentials.atomic_potential import AtomicPotential
from jaxip.potentials.settings import NeuralNetworkPotentialSettings as Settings
from jaxip.structure.structure import Structure
from jaxip.types import Array, Element
from jaxip.types import dtype as default_dtype


def tree_unflatten(state_vector: Array, tree_shape: Dict) -> Dict[Element, frozendict]:
    # FIXME: use tree_map instead of direct looping over the tree items
    global_index: int = 0
    dict_: Dict[Element, defaultdict] = defaultdict(lambda: defaultdict(dict))
    for element in tree_shape:
        for layer in tree_shape[element]:
            for weight_type in tree_shape[element][layer]:
                shape = tree_shape[element][layer][weight_type]
                assert isinstance(
                    shape, tuple
                ), "expected shape tuple[int, ...] for each item in the tree"
                size: int = prod(shape)
                dict_[element][layer][weight_type] = state_vector[
                    global_index : global_index + size, ...
                ].reshape(shape)
                global_index += size
    return flax.core.frozen_dict.FrozenDict(dict_)  # type: ignore


def tree_flatten(params, shape: Tuple[int, int] = (-1, 1), axis: int = 0) -> Array:
    # TODO: investigate jax.flatten_util.ravel_pytree(model_params)
    var: List[Array] = list()
    jax.tree_util.tree_map(lambda x: var.append(x.reshape(shape)), params)
    return jnp.concatenate(var, axis=axis)


class Potential(Protocol):
    settings: Settings
    model_params: Dict[Element, frozendict]
    atomic_potential: Dict[Element, AtomicPotential]


class KalmanFilterTrainer:
    """
    Potential training which uses Kalman filter to update trainable weights (see this paper_).

    .. _paper: https://github.com/CompPhysVienna/n2p2/blob/master/src/libnnptrain/KalmanFilter.cpp
    """

    def __init__(self, potential: Potential) -> None:
        self.potential: Potential = potential
        self._init_parameters()
        self._init_matrices()

    def _init_parameters(self) -> None:
        """Set required parameters from the potential settings."""
        settings: Settings = self.potential.settings

        self.kalman_type: int = settings.kalman_type
        self.beta: float = settings.force_weight
        self.force_fraction: float = settings.short_force_fraction
        # Standard
        self.epsilon: float = settings.kalman_epsilon
        self.q: float = settings.kalman_q0
        self.q_tau: float = settings.kalman_qtau
        self.q_min: float = settings.kalman_qmin
        self.eta: float = settings.kalman_eta
        self.eta_tau: float = settings.kalman_eta
        self.eta_max: float = settings.kalman_etamax

        # Fading memory
        if self.kalman_type == 1:
            self.lambda0: float = settings.kalman_lambda_short
            self.nu: float = settings.kalman_neu_short
            self.gamma: float = 1.0

    def _init_matrices(self) -> None:
        """Initialize required matrices."""
        model_params: Dict[str, frozendict] = self.potential.model_params
        # TODO: add dtype

        # state vector
        self.W: Array = tree_flatten(model_params)
        self.num_states: int = self.W.shape[0]
        # Error covariance matrix
        self.P: Array = (1.0 / self.epsilon) * jnp.identity(
            self.num_states, dtype=default_dtype.FLOATX
        )
        # This will be used to reconstruct model_params dict from the state vector
        self.tree_shape: Dict[Element, frozendict] = jax.tree_util.tree_map(
            lambda x: x.shape, model_params
        )

    def train(
        self,
        dataset: StructureDataset,
    ) -> defaultdict:
        """
        Train potential weights.

        :param dataset: training dataset
        :type dataset: StructureDataset
        """
        settings: Settings = self.potential.settings

        atomic_potential: Dict[
            Element, AtomicPotential
        ] = self.potential.atomic_potential

        model_params: Dict[Element, frozendict] = self.potential.model_params

        def compute_energy_error(
            model_params: Dict[Element, frozendict], structure: Structure
        ) -> Array:
            E_ref: Array = structure.total_energy
            E_pot: Array = _compute_energy(
                frozendict(atomic_potential),
                structure.get_positions(),
                model_params,
                structure.get_inputs(),
            )
            return (E_ref - E_pot)[0]

        # ----------------------

        def compute_force_error(
            state_vector: Array, structure: Structure, tree_shape: Dict
        ) -> Array:
            model_params = tree_unflatten(state_vector, tree_shape)
            F_ref: Array = tree_flatten(structure.get_forces())
            F_pot: Array = tree_flatten(
                _compute_force(
                    frozendict(atomic_potential),
                    structure.get_positions(),
                    model_params,
                    structure.get_inputs(),
                )
            )
            return (F_ref - F_pot)[..., 0]

        grad_energy_error: Callable = jax.grad(compute_energy_error)
        jacob_force_error: Callable = jax.jacfwd(compute_force_error)

        def compute_energy_error_gradient(
            model_params: Dict[Element, frozendict], structure: Structure
        ) -> Array:
            return tree_flatten(
                grad_energy_error(model_params, structure),
            )

        def compute_force_error_jacobian(
            state_vector: Array, structure: Structure, tree_shape: Dict
        ) -> Array:
            return jacob_force_error(
                state_vector[..., 0], structure, tree_shape
            ).transpose()

        # ----------------------

        # FIXME: dataloader
        indices: list[int] = [i for i in range(len(dataset))]

        history = defaultdict(list)
        for epoch in range(settings.epochs):

            print(f"Epoch: {epoch}")
            random.shuffle(indices)
            loss_per_epoch: Array = jnp.asarray(0.0)
            num_update_per_epoch: int = 0

            for index in tqdm(indices):

                structure: Structure = dataset[index]

                # Error and jacobian matrices
                if np.random.rand() < self.force_fraction:
                    Xi = self.beta * compute_force_error(
                        self.W,
                        structure,
                        self.tree_shape,
                    ).reshape(-1, 1)
                    H = -self.beta * compute_force_error_jacobian(
                        self.W,
                        structure,
                        self.tree_shape,
                    )
                else:
                    Xi = compute_energy_error(model_params, structure).reshape(-1, 1)
                    H = -compute_energy_error_gradient(model_params, structure)

                num_observations: int = Xi.shape[0]

                # A temporary matrix
                X = self.P @ H

                # Scaling matrix
                A = H.transpose() @ X

                # Update learning rate
                if (self.kalman_type == 0) and (self.eta < self.eta_max):
                    self.eta *= np.exp(self.eta_tau)

                # Measurement noise
                if self.kalman_type == 0:
                    A += (1.0 / self.eta) * jnp.identity(
                        num_observations, dtype=default_dtype.FLOATX
                    )
                elif self.kalman_type == 1:
                    A += self.lambda0 * jnp.identity(
                        num_observations, dtype=default_dtype.FLOATX
                    )

                # Kalman gain matrix
                K = X @ jnp.linalg.inv(A)

                # Update error covariance matrix
                self.P -= K @ X.transpose()

                # Forgetting factor
                if self.kalman_type == 1:
                    self.P *= 1.0 / self.lambda0

                # Process noise.
                self.P += self.q * jnp.identity(
                    self.num_states, dtype=default_dtype.FLOATX
                )

                # Update state vector
                self.W += K @ Xi

                # Anneal process noise
                if self.q > self.q_min:
                    self.q *= np.exp(-self.q_tau)

                # Update forgetting factor
                if self.kalman_type == 1:
                    self.lambda0 = self.nu * self.lambda0 + 1.0 - self.nu
                    self.gamma = 1.0 / (1.0 + self.lambda0 / self.gamma)

                # Get params from state vector
                model_params = tree_unflatten(self.W, self.tree_shape)

                loss_per_epoch += jnp.matmul(Xi.transpose(), Xi)[0, 0]
                num_update_per_epoch += 1

            # Update model params
            logger.debug(f"Updating potential weights after epoch {epoch + 1}")
            self.potential.model_params = model_params

            loss_per_epoch /= num_update_per_epoch
            print(f"loss: {loss_per_epoch}")
            history["epoch"].append(epoch + 1)
            history["loss"].append(loss_per_epoch)

        return history
