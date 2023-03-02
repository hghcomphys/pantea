from __future__ import annotations

import math
import random
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List, Protocol, Tuple

import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from frozendict import frozendict
from jax import jit, value_and_grad
from tqdm import tqdm

from jaxip.datasets.base import StructureDataset
from jaxip.logger import logger
from jaxip.potentials._energy import _energy_fn
from jaxip.potentials._force import _compute_force
from jaxip.potentials.atomic_potential import AtomicPotential
from jaxip.potentials.loss import mse_loss
from jaxip.potentials.metrics import ErrorMetric
from jaxip.potentials.settings import NeuralNetworkPotentialSettings as Settings
from jaxip.structure.structure import Structure
from jaxip.types import Array, Element


class Potential(Protocol):
    settings: Settings
    elements: List[Element]
    atomic_potential: Dict[Element, AtomicPotential]
    model_params: Dict[Element, frozendict]


# @dataclass
class GradientDescentTrainer:
    """
    A trainer class to fit a generic NNP potential using target values of the total
    energy and force components.

    See https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    """

    # potential: PotentialInterface
    # settings: Settings
    # criterion: Callable = field(default_factory=lambda: mse_loss)
    # error_metric: ErrorMetric
    # optimizer: Dict[Element, Any]

    def __init__(self, potential) -> None:

        self.potential = potential
        self.settings: Settings = potential.settings
        self.criterion: Callable[..., Array] = mse_loss
        self.error_metric: ErrorMetric = ErrorMetric.create_from(
            self.settings.main_error_metric
        )
        self.optimizer = self._init_optimizer()

    def _init_optimizer(self) -> Any:
        """Create optimizer using the potential settings."""

        if self.settings.updater_type == "gradient_descent":
            if self.settings.gradient_type == "adam":
                optimizer_cls: Callable = optax.adamw
                optimizer_kwargs: Dict[str, Any] = {
                    "learning_rate": self.settings.gradient_adam_eta,
                    "b1": self.settings.gradient_adam_beta1,
                    "b2": self.settings.gradient_adam_beta2,
                    "eps": self.settings.gradient_adam_epsilon,
                    "weight_decay": self.settings.gradient_adam_weight_decay,
                }
            # TODO: self.settings.gradient_type == "fixed_step":
            else:
                logger.error(
                    f"Gradient type {self.settings.gradient_type} is not implemented yet",
                    exception=NotImplementedError,
                )
        else:
            logger.error(
                f"Unknown updater type {self.settings.updater_type}",
                exception=NotImplementedError,
            )
        # TODO: either as a single but global optimizer or multiple optimizers:
        optimizer = optimizer_cls(**optimizer_kwargs)  # type: ignore
        # return {element: optimizer for element in self.elements}
        return optimizer

    def _init_train_state(self) -> Dict[Element, TrainState]:
        """Initialize train state for each element."""

        def generate_train_state():
            for element in self.potential.elements:
                yield element, TrainState.create(
                    apply_fn=self.potential.atomic_potential[element].model.apply,
                    params=self.potential.model_params[element],
                    tx=self.optimizer,  # [element]?
                )

        return {element: state for element, state in generate_train_state()}

    # ------------------------------------------------------------------------

    def fit(self, dataset: StructureDataset, **kwargs):
        """Train potential."""
        # TODO: add writer and tensorboard
        # TODO: add data loader: batch_size, shuffle, train/val split, etc.

        batch_size: int = kwargs.get("batch_size", 1)
        steps: int = kwargs.get("steps", math.ceil(len(dataset) / batch_size))
        epochs: int = kwargs.get("epochs", 50)

        states: Dict[Element, TrainState] = self._init_train_state()
        history = defaultdict(list)

        # Loop over epochs
        for epoch in range(epochs):

            print(f"[Epoch {epoch+1} of {epochs}]")
            history["epoch"].append(epoch)

            # Loop over batch of structures
            for _ in tqdm(range(steps)):

                structures: List[Structure] = random.choices(dataset, k=batch_size)
                xbatch = tuple(structure.get_inputs() for structure in structures)
                ybatch = tuple(
                    (structure.total_energy, structure.get_forces())
                    for structure in structures
                )

                batch = xbatch, ybatch
                states, metrics = self.train_step(states, batch)

                self._update_model_params(states)

            print(
                f"training loss:{float(metrics['loss']): 0.7f}"
                # f", loss[energy]:{float(metrics['loss_eng']): 0.7f}"
                # f", loss[force]:{float(metrics['loss_frc']): 0.7f}"
                # f"\n"
            )
            history["loss"].append(metrics["loss"])

        return history

    @partial(jit, static_argnums=(0,))  # FIXME
    def train_step(
        self,
        states: Dict[Element, TrainState],
        batch: Tuple,
    ) -> Tuple:
        """Train potential on a batch of data."""

        def loss_fn(params: Dict[Element, frozendict]) -> Tuple[Array, Any]:
            """Loss function."""
            # TODO: define force loss weights for each element
            xbatch, ybatch = batch
            batch_size = len(xbatch)
            loss = jnp.array(0.0)
            for inputs, (true_energy, true_forces) in zip(xbatch, ybatch):

                positions = {
                    element: input.atom_position for element, input in inputs.items()
                }
                natoms: int = sum(array.shape[0] for array in positions.values())
                # ------ energy ------
                energy = _energy_fn(
                    frozendict(self.potential.atomic_potential),
                    positions,
                    params,
                    inputs,
                )
                loss_eng = self.criterion(logits=energy, targets=true_energy) / natoms
                loss += loss_eng
                # if random.random() < 0.15:
                # ------ Force ------
                # forces = _compute_force(
                #     frozendict(self.potential.atomic_potential),
                #     positions,
                #     params,
                #     inputs,
                # )
                # elements = true_forces.keys()
                # loss_frc = jnp.array(0.0)
                # for element in elements:
                #     loss_frc += self.criterion(
                #         logits=forces[element],
                #         targets=true_forces[element],
                #     )
                # loss_frc /= len(forces)
                # loss += loss_frc

            loss /= batch_size
            return loss, (jnp.array(0),)  # (loss_eng, loss_frc) #, (energy, forces))

        value_and_grad_fn = value_and_grad(loss_fn, has_aux=True)

        (loss, (_,)), grads = value_and_grad_fn(
            {element: state.params for element, state in states.items()}
        )
        states = {
            element: states[element].apply_gradients(grads=grads[element])
            for element in states.keys()
        }

        # TODO: add more metrics for force
        metrics = {
            "loss": loss,
            # "loss_eng": loss_eng,
            # "loss_frc": loss_frc,
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
