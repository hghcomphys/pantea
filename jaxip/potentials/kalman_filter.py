import random
from collections import defaultdict
from typing import Callable, Dict, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from frozendict import frozendict
from jax import flatten_util
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


def _tree_flatten(pytree: Dict) -> Array:
    return flatten_util.ravel_pytree(pytree)[0].reshape(-1, 1)  # type: ignore


class Potential(Protocol):
    settings: Settings
    model_params: Dict[Element, frozendict]
    atomic_potential: Dict[Element, AtomicPotential]


class KalmanFilterTrainer:
    """
    Potential training which uses Kalman filter to update trainable weights (see this_).

    .. _this: https://pubs.acs.org/doi/10.1021/acs.jctc.8b01092
    """

    # https://github.com/CompPhysVienna/n2p2/blob/master/src/libnnptrain/KalmanFilter.cpp

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

        # Initialize state vector
        W, tree_unflatten = flatten_util.ravel_pytree(model_params)  # type: ignore
        self.W: Array = W.reshape(-1, 1)
        self._unflatten_state_vector: Callable = tree_unflatten
        self.num_states: int = self.W.shape[0]
        # Error covariance matrix
        self.P: Array = (1.0 / self.epsilon) * jnp.identity(
            self.num_states, dtype=default_dtype.FLOATX
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

        # ----------------------

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

        def compute_force_error(state_vector: Array, structure: Structure) -> Array:
            model_params: Dict = self._unflatten_state_vector(state_vector)
            F_ref: Array = _tree_flatten(structure.get_forces())
            F_pot: Array = _tree_flatten(
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
            return _tree_flatten(
                grad_energy_error(model_params, structure),
            )

        def compute_force_error_jacobian(
            state_vector: Array, structure: Structure
        ) -> Array:
            return jacob_force_error(state_vector[..., 0], structure).transpose()

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
                    ).reshape(-1, 1)
                    H = -self.beta * compute_force_error_jacobian(
                        self.W,
                        structure,
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
                model_params = self._unflatten_state_vector(self.W)

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
