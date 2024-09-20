from __future__ import annotations

import random
from collections import defaultdict
from ctypes import Structure
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, NamedTuple, Protocol, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import flatten_util
from tqdm import tqdm

from pantea.atoms.structure import StructureAsKernelArgs
from pantea.datasets.dataset import Dataset
from pantea.descriptors.scaler import ScalerParams
from pantea.logger import logger
from pantea.models.nn.model import ModelParams
from pantea.potentials.nnp.atomic_potential import AtomicPotential
from pantea.potentials.nnp.energy import _compute_energy
from pantea.potentials.nnp.force import _compute_forces
from pantea.potentials.nnp.potential import NeuralNetworkPotential
from pantea.potentials.nnp.settings import NeuralNetworkPotentialSettings
from pantea.types import Array, Element, default_dtype


class TrainingParamsInterface(Protocol):
    force_weight: float
    energy_fraction: float
    force_fraction: float
    epochs: int


class KernelCommonArgs(NamedTuple):
    atomic_potentials: Dict[Element, AtomicPotential]
    positions: Dict[Element, Array]
    scalers_params: Dict[Element, ScalerParams]
    structure: StructureAsKernelArgs


@dataclass
class KalmanFilter:
    """
    Kalman Filter implementation for updating model parameters to predict
    the total potential energy and force components (gradients).

    See  https://pubs.acs.org/doi/10.1021/acs.jctc.8b01092
    Also https://github.com/CompPhysVienna/n2p2/blob/master/src/libnnptrain/KalmanFilter.cpp
    """

    kalman_type: int
    epsilon: float
    q: float
    q_tau: float
    q_min: float
    eta: float
    eta_tau: float
    eta_max: float
    # Fading memory
    lambda0: float
    neu: float
    gamma: float
    # state
    num_states: int
    W: Array = field(repr=False)
    P: Array = field(repr=False)
    unflatten_state_vector: Callable = field(repr=False)
    potential: NeuralNetworkPotential = field(repr=False)

    @classmethod
    def from_runner(
        cls,
        potential: NeuralNetworkPotential,
        filename: str = "input.nn",
    ):
        potfile = potential.directory / filename
        settings = NeuralNetworkPotentialSettings.from_file(potfile)
        models_params = potential.models_params
        epsilon = settings.kalman_epsilon

        # Setting random seed
        random.seed(settings.random_seed)
        np.random.seed(settings.random_seed)

        # Initialize state vector
        W, tree_unflatten = flatten_util.ravel_pytree(models_params)  # type: ignore
        W_vector = W.reshape(-1, 1)
        num_states = W_vector.shape[0]
        # Error covariance matrix
        P = (1.0 / epsilon) * jnp.identity(num_states, dtype=default_dtype.FLOATX)

        return cls(
            kalman_type=settings.kalman_type,
            epsilon=settings.kalman_epsilon,
            q=settings.kalman_q0,
            q_tau=settings.kalman_qtau,
            q_min=settings.kalman_qmin,
            eta=settings.kalman_eta,
            eta_tau=settings.kalman_eta,
            eta_max=settings.kalman_etamax,
            # Fading memory
            lambda0=settings.kalman_lambda_short,
            neu=settings.kalman_neu_short,
            gamma=1.0,
            W=W_vector,
            P=P,
            unflatten_state_vector=tree_unflatten,
            num_states=num_states,
            potential=potential,
        )

    def fit(
        self,
        dataset: Dataset,
        training_params: TrainingParamsInterface,
    ) -> defaultdict[str, Any]:
        """Fit the potential model parameters."""

        atomic_potentials = self.potential.atomic_potentials
        models_params = self.potential.models_params
        scalers_params = self.potential.scalers_params

        indices: list[int] = [i for i in range(len(dataset))]

        history = defaultdict(list)
        for epoch in range(training_params.epochs):
            print(f"Epoch: {epoch + 1} of {training_params.epochs}")
            random.shuffle(indices)

            loss_energy_per_epoch = 0.0
            loss_force_per_epoch = 0.0
            num_energy_updates_per_epoch: int = 0
            num_force_updates_per_epoch: int = 0

            for index in tqdm(indices):

                structure = dataset[index]
                kernel_common_args = KernelCommonArgs(
                    atomic_potentials,
                    structure.get_positions_per_element(),
                    scalers_params,
                    structure.as_kernel_args(),
                )

                # Error and jacobian matrices
                if np.random.rand() < training_params.force_fraction:
                    loss, (Xi, H) = self.calculate_loss_force(
                        kernel_common_args,
                        structure.get_forces_per_element(),
                        self.unflatten_state_vector,
                        self.W,
                    )
                    loss_force_per_epoch += loss
                    num_force_updates_per_epoch += 1
                else:
                    loss, (Xi, H) = self.calculate_loss_energy(
                        kernel_common_args,
                        models_params,
                    )
                    loss_energy_per_epoch += loss
                    num_energy_updates_per_epoch += 1

                self.update_state_vector(Xi, H)
                models_params = self.unflatten_state_vector(self.W)

            # Update model params
            logger.debug(f"Updating potential weights after epoch {epoch + 1}")
            self.potential.models_params = models_params

            loss_energy_per_epoch /= (
                num_energy_updates_per_epoch if num_energy_updates_per_epoch > 0 else 1
            )
            loss_force_per_epoch /= (
                num_force_updates_per_epoch if num_force_updates_per_epoch > 0 else 1
            )
            loss_per_epoch = loss_energy_per_epoch + loss_force_per_epoch
            num_updates_per_epoch = (
                num_energy_updates_per_epoch + num_force_updates_per_epoch
            )

            print(
                f"training loss:{float(loss_per_epoch): 0.7f}"
                f", loss_energy:{float(loss_energy_per_epoch): 0.7f}"
                f", loss_force:{float(loss_force_per_epoch): 0.7f}"
            )
            history["epoch"].append(epoch + 1)
            history["loss"].append(loss_per_epoch)

            logger.info(
                f"energy_update_ratio: {num_energy_updates_per_epoch/num_updates_per_epoch:.3f}"
                f", force_update_ratio: {num_force_updates_per_epoch/num_updates_per_epoch:.3f}"
            )
        return history

    def update_state_vector(self, Xi: Array, H: Array) -> None:
        """Update state vector using energy (force) error and its gradient."""
        # A temporary matrix
        X = self.P @ H
        # Scaling matrix
        A = H.transpose() @ X
        # Update learning rate
        if (self.kalman_type == 0) and (self.eta < self.eta_max):
            self.eta *= np.exp(self.eta_tau)
        # Measurement noise
        num_observations = Xi.shape[0]
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
        self.P += self.q * jnp.identity(self.num_states, dtype=default_dtype.FLOATX)
        # Update state vector
        self.W += K @ Xi
        # Anneal process noise
        if self.q > self.q_min:
            self.q *= np.exp(-self.q_tau)
        # Update forgetting factor
        if self.kalman_type == 1:
            self.lambda0 = self.neu * self.lambda0 + 1.0 - self.neu
            self.gamma = 1.0 / (1.0 + self.lambda0 / self.gamma)

    @classmethod
    def calculate_loss_energy(
        cls,
        kernel_common_args: KernelCommonArgs,
        models_params: Dict[Element, ModelParams],
    ) -> Tuple[Array, Tuple[Array, Array]]:
        kernel_args = (*kernel_common_args, models_params)
        # energy error
        Xi = _jitted_compute_energy_error(*kernel_args).reshape(-1, 1)
        loss = jnp.matmul(Xi.transpose(), Xi)[0, 0]
        # gradient of energy error
        gradients = _jitted_grad_compute_energy_error(*kernel_args)
        H = -1.0 * _tree_flatten(gradients)
        return loss, (Xi, H)

    @classmethod
    def calculate_loss_force(
        cls,
        kernel_common_args: KernelCommonArgs,
        forces: Dict[Element, Array],
        unflatten_state_vector: Callable,
        W: Array,
    ) -> Tuple[Array, Tuple[Array, Array]]:
        kernel_args = (*kernel_common_args, forces, unflatten_state_vector, W[..., 0])
        # force error
        Xi = _jitted_compute_forces_error(*kernel_args).reshape(-1, 1)
        loss = jnp.matmul(Xi.transpose(), Xi)[0, 0]
        # gradient of force error
        gradients = _jitted_grad_compute_forces_error(
            *kernel_common_args,
            forces,
            unflatten_state_vector,
            W[..., 0],
        )
        H = -1.0 * gradients.transpose()
        return loss, (Xi, H)


def _tree_flatten(pytree: Dict) -> Array:
    return flatten_util.ravel_pytree(pytree)[0].reshape(-1, 1)  # type: ignore


def _compute_energy_error(
    atomic_potentials: Dict[Element, AtomicPotential],
    positions: Dict[Element, Array],
    scalers_params: Dict[Element, ScalerParams],
    structure: StructureAsKernelArgs,
    models_params: Dict[Element, ModelParams],
) -> Array:
    natoms = structure.positions.shape[0]
    E_ref = structure.total_energy
    E_pot = _compute_energy(
        atomic_potentials,
        positions,
        models_params,
        scalers_params,
        structure,
    )
    return (E_ref - E_pot) / natoms


_jitted_compute_energy_error = jax.jit(_compute_energy_error, static_argnums=(0,))


_grad_compute_energy_error = jax.grad(_compute_energy_error, argnums=4)

_jitted_grad_compute_energy_error = jax.jit(
    _grad_compute_energy_error, static_argnums=(0,)
)


def _compute_forces_error(
    atomic_potentials: Dict[Element, AtomicPotential],
    positions: Dict[Element, Array],
    scalers_params: Dict[Element, ScalerParams],
    structure: StructureAsKernelArgs,
    forces: Dict[Element, Array],
    unflatten_state_vector: Callable,
    state_vector: Array,
) -> Array:
    models_params = unflatten_state_vector(state_vector)
    F_ref = _tree_flatten(forces)
    F_pot = _tree_flatten(
        _compute_forces(
            atomic_potentials,
            positions,
            models_params,
            scalers_params,
            structure,
        )
    )
    return (F_ref - F_pot)[..., 0]


_jitted_compute_forces_error = jax.jit(_compute_forces_error, static_argnums=(0, 5))


_grad_compute_forces_error = jax.jacrev(_compute_forces_error, argnums=6)

_jitted_grad_compute_forces_error = jax.jit(
    _grad_compute_forces_error, static_argnums=(0, 5)
)
