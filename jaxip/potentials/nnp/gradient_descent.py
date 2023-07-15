from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from frozendict import frozendict
from jax import value_and_grad
from optax import GradientTransformation
from tqdm import tqdm

from jaxip.atoms.structure import Structure
from jaxip.datasets.dataset import DatasetInterface
from jaxip.logger import logger
from jaxip.potentials._energy import _energy_fn
from jaxip.potentials._force import _compute_force
from jaxip.potentials.nnp.metrics import ErrorMetric
from jaxip.potentials.nnp.nnp import (
    NeuralNetworkPotentialInterface as PotentialInterface,
)
from jaxip.potentials.nnp.settings import (
    NeuralNetworkPotentialSettings as PotentialSettings,
)
from jaxip.types import Array, Element


def _mse_loss(*, logits: Array, targets: Array) -> Array:
    return ((targets - logits) ** 2).mean()


# @dataclass
class GradientDescentUpdater:
    """
    A trainer class to fit a generic NNP potential weights using target values of the total
    energy and force components.

    See https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    """

    # potential: PotentialInterface
    # settings: PotentialSettings
    # criterion: Callable = field(default_factory=lambda: mse_loss)
    # error_metric: ErrorMetric
    # optimizer: Dict[Element, Any]

    def __init__(self, potential: PotentialInterface) -> None:
        """Initialize potential."""
        self.potential: PotentialInterface = potential
        self.criterion: Callable[..., Array] = _mse_loss
        self.error_metric: ErrorMetric = ErrorMetric.create(
            self.potential.settings.main_error_metric
        )
        self._init_parameters()
        self._init_optimizer()

    def _init_parameters(self) -> None:
        """Set required parameters from the potential settings."""
        settings: PotentialSettings = self.potential.settings
        self.gradient_type: str = settings.gradient_type
        self.beta: float = settings.force_weight
        self.force_fraction: float = settings.short_force_fraction

    def _init_optimizer(self) -> None:
        """Create optimizer using the potential settings."""
        settings: PotentialSettings = self.potential.settings
        if self.gradient_type == "adam":
            self.optimizer: GradientTransformation = optax.adamw(
                learning_rate=settings.gradient_adam_eta,
                b1=settings.gradient_adam_beta1,
                b2=settings.gradient_adam_beta2,
                eps=settings.gradient_adam_epsilon,
                weight_decay=settings.gradient_adam_weight_decay,
            )
        else:  # TODO: self.settings.gradient_type == "fixed_step":
            logger.error(
                f"Gradient type {settings.gradient_type} is not implemented yet",
                exception=TypeError,
            )
        # Single global optimizer or multiple optimizers:
        # return {element: optimizer for element in self.elements}

    def _init_train_state(self) -> Dict[Element, TrainState]:
        """Initialize train state for each element."""

        def generate_train_state():
            for element in self.potential.elements:
                yield element, TrainState.create(
                    apply_fn=self.potential.atomic_potential[
                        element
                    ].model.apply,
                    params=self.potential.model_params[element],
                    tx=self.optimizer,  # [element]?
                )

        return {element: state for element, state in generate_train_state()}

    # ------------------------------------------------------------------------

    def fit(self, dataset: DatasetInterface, **kwargs):
        """Train potential."""
        batch_size: int = kwargs.get("batch_size", 1)
        steps: int = kwargs.get("steps", math.ceil(len(dataset) / batch_size))
        epochs: int = kwargs.get("epochs", 10)

        states: Dict[Element, TrainState] = self._init_train_state()
        history = defaultdict(list)

        # Loop over epochs
        for epoch in range(epochs):
            print(f"[Epoch {epoch+1} of {epochs}]")

            loss_per_epoch: Array = jnp.array(0.0)
            loss_energy_per_epoch: Array = jnp.array(0.0)
            loss_force_per_epoch: Array = jnp.array(0.0)
            num_updates_per_epoch: int = 0

            # Loop over batches
            for _ in tqdm(range(steps)):
                structures: List[Structure] = random.choices(
                    dataset, k=batch_size
                )
                xbatch = tuple(
                    structure.get_per_element_inputs()
                    for structure in structures
                )
                ybatch = tuple(
                    (
                        structure.total_energy,
                        structure.get_per_element_forces(),
                    )
                    for structure in structures
                )

                batch = xbatch, ybatch
                states, metrics = self.train_step(states, batch)

                loss_per_epoch += metrics["loss"]
                loss_energy_per_epoch += metrics["loss_energy"]
                loss_force_per_epoch += metrics["loss_force"]
                num_updates_per_epoch += 1

            loss_per_epoch /= num_updates_per_epoch
            loss_energy_per_epoch /= num_updates_per_epoch
            loss_force_per_epoch /= num_updates_per_epoch

            self._update_model_params(states)

            print(
                f"training loss:{float(loss_per_epoch): 0.7f}"
                f", loss_energy:{float(loss_energy_per_epoch): 0.7f}"
                f", loss_force:{float(loss_force_per_epoch): 0.7f}"
            )
            history["epoch"].append(epoch)
            history["loss"].append(loss_per_epoch)

        return history

    # @partial(jit, static_argnums=(0,))
    def train_step(
        self,
        states: Dict[Element, TrainState],
        batch: Tuple,
    ) -> Tuple:
        """Train potential on a batch of data."""

        def loss_fn(params: Dict[Element, frozendict]) -> Tuple[Array, Any]:
            """Loss function."""
            xbatch, ybatch = batch
            batch_size = len(xbatch)

            loss_energy_per_batch: Array = jnp.array(0.0)
            loss_force_per_batch: Array = jnp.array(0.0)

            for inputs, (true_energy, true_forces) in zip(xbatch, ybatch):
                positions = {
                    element: input.atom_positions
                    for element, input in inputs.items()
                }
                natoms: int = sum(
                    array.shape[0] for array in positions.values()
                )

                if np.random.rand() < self.force_fraction:
                    # ------ Force ------
                    forces = _compute_force(
                        frozendict(self.potential.atomic_potential),
                        positions,
                        params,
                        inputs,
                    )
                    elements = true_forces.keys()
                    loss_force = jnp.array(0.0)
                    for element in elements:
                        loss_force += self.criterion(
                            logits=forces[element],
                            targets=true_forces[element],
                        )
                    loss_force /= len(forces)
                    loss_force_per_batch += self.beta * self.beta * loss_force

                else:
                    # ------ energy ------
                    energy = _energy_fn(
                        frozendict(self.potential.atomic_potential),
                        positions,
                        params,
                        inputs,
                    )
                    loss_energy = (
                        self.criterion(logits=energy, targets=true_energy)
                        / natoms
                    )
                    loss_energy_per_batch += loss_energy

            loss_energy_per_batch /= batch_size
            loss_force_per_batch /= batch_size
            loss_per_batch = loss_energy_per_batch + loss_force_per_batch

            return loss_per_batch, (
                loss_energy_per_batch,
                loss_force_per_batch,
            )

        value_and_grad_fn = value_and_grad(loss_fn, has_aux=True)

        (loss, (loss_energy, loss_force)), grads = value_and_grad_fn(
            {element: state.params for element, state in states.items()}
        )
        states = {
            element: states[element].apply_gradients(grads=grads[element])
            for element in states.keys()
        }

        metrics = {
            "loss": loss,
            "loss_energy": loss_energy,
            "loss_force": loss_force,
        }
        return states, metrics

    def _update_model_params(self, states: Dict[Element, TrainState]) -> None:
        """Update model params for all the elements."""
        for element, train_state in states.items():
            self.potential.model_params[element] = train_state.params  # type: ignore

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\npotential={self.potential}, \noptimizer={self.optimizer}, "
            f"\ncriterion={self.criterion}, \nerror_metric={self.error_metric})"
        )
