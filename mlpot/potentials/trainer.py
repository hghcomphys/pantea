from ..logger import logger
from ..potentials.base import Potential
from ..datasets.base import StructureDataset
from ..descriptors.asf._asf import _calculate_descriptor
from ..base import _Base
from .loss import mse_loss
from .metrics import ErrorMetric
from .metrics import init_error_metric
from collections import defaultdict
from typing import Dict, Callable, Tuple, Generator
from flax.training.train_state import TrainState
from frozendict import frozendict
from functools import partial
from jax import jit, grad
import jax.numpy as jnp
import optax


@partial(jit, static_argnums=(0,))  # FIXME
def _energy_fn(
    sargs: Tuple,
    xs: Tuple[jnp.ndarray],
    state: Tuple[TrainState],
    params: frozendict,
    xbatch: Tuple[jnp.ndarray],
) -> jnp.ndarray:
    """
    A helper function that allows to calculate gradient of the NNP total energy
    respect to the atom positions (for each element).
    """
    energy = jnp.array(0.0)
    for s, p, inputs, static_inputs, x in zip(state, params, xbatch, sargs, xs):
        _, position, atype, lattice, emap = inputs
        asf, scaler, dtype = static_inputs

        dsc = _calculate_descriptor(asf, x, position, atype, lattice, dtype, emap)
        scaled_dsc = scaler(dsc)
        energy += jnp.sum(s.apply_fn({"params": p}, scaled_dsc))

    return energy


_grad_energy_fn = jit(
    grad(_energy_fn, argnums=1),
    static_argnums=0,
)


class NeuralNetworkPotentialTrainer(_Base):
    """
    A trainer class to fit a generic NNP potential using target values of the total
    energy and force components.

    See https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    """

    def __init__(self, potential: Potential, **kwargs) -> None:
        """
        Initialize trainer.
        """
        self.potential = potential
        self.save_best_model: bool = kwargs.get("save_best_model", True)

        self.criterion: Callable = mse_loss
        self.error_metric: ErrorMetric = init_error_metric(
            self.potential.settings["main_error_metric"]
        )

        self.optimizer = self.init_optimizer()

        # self.force_weight: float = self.potential.settings["force_weight"]
        # self.atom_energy: Dict[str, float] = self.potential.settings["atom_energy"]

    def init_optimizer(self) -> Dict:
        """
        Prepare optimizer using potential settings.

        :return: optimizer
        :rtype: torch.optim.Optimizer
        """
        settings = self.potential.settings

        if settings["updater_type"] == 0:  # Gradient Descent

            if settings["gradient_type"] == 1:  # Adam
                optimizer_cls = optax.adam
                optimizer_cls_kwargs = {
                    "learning_rate": settings["gradient_adam_eta"],
                    "b1": settings["gradient_adam_beta1"],
                    "b2": settings["gradient_adam_beta2"],
                    "eps": settings["gradient_adam_epsilon"],
                    # "weight_decay": self.settings["gradient_weight_decay"], # TODO: regularization?
                }
            # TODO: self.settings["gradient_type"] == 0:  # fixed Step
            else:
                logger.error(
                    f'Gradient type {settings["gradient_type"]} is not implemented yet',
                    exception=NotImplementedError,
                )
        else:
            logger.error(
                f'Unknown updater type {settings["updater_type"]}',
                exception=NotImplementedError,
            )

        # TODO: either as a single but global optimizer or multiple optimizers:
        optimizer = optimizer_cls(**optimizer_cls_kwargs)

        # return {element: optimizer for element in self.elements}
        return optimizer

    def init_train_state(self) -> Tuple[TrainState]:
        """
        Create train state for each element.
        """

        def generate_train_state():
            for element in self.potential.elements:
                yield TrainState.create(
                    apply_fn=self.potential.model[element].apply,
                    params=self.potential.model_params[element],
                    tx=self.optimizer,  # [element]?
                )

        return tuple(state for state in generate_train_state())

    def eval_step(self, batch):
        pass

    @partial(jit, static_argnums=(0, 1))  # FIXME
    def train_step_energy(
        self,
        sargs: Tuple,
        state: Tuple[TrainState],
        batch: Tuple[Tuple, Tuple[jnp.ndarray]],
    ) -> Tuple:

        xbatch, target_energy = batch

        positions: Tuple[jnp.ndarray] = tuple(inputs[0] for inputs in xbatch)
        n_atoms: int = sum(pos.shape[0] for pos in positions)

        def loss_fn(params: Tuple[frozendict]) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """
            Loss function based on energy.
            """
            energy = _energy_fn(sargs, positions, state, params, xbatch)
            loss = self.criterion(logits=energy, targets=target_energy) / n_atoms

            return loss, energy

        grad_fn = grad(loss_fn, has_aux=True)
        grads, energy = grad_fn(tuple(s.params for s in state))

        state = tuple(s.apply_gradients(grads=g) for s, g in zip(state, grads))
        metrics = mse_loss(logits=energy, targets=target_energy) / n_atoms

        return state, metrics

    @partial(jit, static_argnums=(0, 1))  # FIXME
    def train_step_force(
        self,
        sargs: Tuple,
        state: Tuple[TrainState],
        batch: Tuple[Tuple, Tuple[jnp.ndarray]],
    ) -> Tuple:

        xbatch, target_forces = batch
        positions: Tuple[jnp.ndarray] = tuple(inputs[0] for inputs in xbatch)

        def loss_fn(
            params: Tuple[frozendict],
        ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray]]:
            """
            Loss function based on force components
            """
            # Calculate forces for each element based on gradient of the total energy
            grad_energies: Tuple[jnp.ndarray] = _grad_energy_fn(
                sargs, positions, state, params, xbatch
            )
            forces: Tuple[jnp.ndarray] = tuple(
                -1 * grad_energy for grad_energy in grad_energies
            )

            # TODO: define loss weights for each element
            loss = jnp.array(0.0)
            for force, target_force in zip(forces, target_forces):
                loss += self.criterion(logits=force, targets=target_force)
            loss /= len(forces)

            return loss, forces

        grad_fn = grad(loss_fn, has_aux=True)
        grads, forces = grad_fn(tuple(s.params for s in state))
        state = tuple(s.apply_gradients(grads=g) for s, g in zip(state, grads))

        metrics = jnp.array(0.0)
        for force, target_force in zip(forces, target_forces):
            metrics += mse_loss(logits=force, targets=target_force)
        metrics /= len(forces)

        return state, metrics

    def fit(self, dataset: StructureDataset, **kwargs):

        # TODO: define as methods
        def extract_inputs(structure) -> Generator:
            for element in self.potential.elements:
                aid = structure.select(element)
                yield (
                    structure.position[aid],
                    structure.position,
                    structure.atype,
                    structure.box.lattice,
                    structure.element_map.element_to_atype,
                )

        def extract_static_inputs(structure) -> Generator:
            for element in self.potential.elements:
                yield (
                    self.potential.descriptor[element],
                    self.potential.scaler[element],
                    structure.dtype,
                )

        def extract_force(structure) -> Generator:
            for element in self.potential.elements:
                aid = structure.select(element)
                yield structure.force[aid]

        history = defaultdict(list)
        state = self.init_train_state()
        for epoch in range(50):

            for structure in dataset:
                sargs = tuple(inp for inp in extract_static_inputs(structure))
                inputs = tuple(inp for inp in extract_inputs(structure))

                batch = inputs, structure.total_energy
                state, metrics = self.train_step_energy(sargs, state, batch)

                batch = inputs, tuple(frc for frc in extract_force(structure))
                state, metrics = self.train_step_force(sargs, state, batch)

            if epoch % 1 == 0:
                print(f"Epoch {epoch}, loss={metrics}")
            history["epoch"].append(epoch)
            history["metrics_train"].append(metrics)

            # update model params
            for element, train_state in zip(self.potential.elements, state):
                self.potential.model_params[element] = train_state.params

        return history

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\npotential={self.potential}, \noptimizer={self.optimizer}, "
            f"\ncriterion={self.criterion}, \nerror_metric={self.error_metric})"
        )
