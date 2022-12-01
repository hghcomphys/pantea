from ..logger import logger
from ..potentials.base import Potential
from ..datasets.base import StructureDataset
from ..descriptors.asf._asf import _calculate_descriptor
from ..base import _Base
from .loss import mse_loss
from .metrics import ErrorMetric
from .metrics import init_error_metric
from collections import defaultdict
from typing import Dict, Callable, Tuple
from flax.training.train_state import TrainState
from frozendict import frozendict
from functools import partial
from jax import jit, grad
import jax.numpy as jnp
import optax


class BasePotentialTrainer(_Base):
    """
    A trainer class for fitting a generic potential.
    This class must be independent of the type of the potential.

    A derived trainer class, which is specific to a potential, can benefit from the best algorithms to
    train the model(s) in the potential using energy and force components.
    """

    pass


class NeuralNetworkPotentialTrainer(BasePotentialTrainer):
    """
    This derived trainer class that trains the neural network potential using energy and force components.

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

    @partial(jit, static_argnums=(0,))  # FIXME
    def train_step_energy(
        self,
        state: Tuple[TrainState],
        batch: Tuple[jnp.ndarray],
    ):
        descriptors, total_energy = batch

        def loss_fn(params: Tuple[frozendict]):
            energy = 0.0
            for s, p, d in zip(state, params, descriptors):
                energy += jnp.sum(s.apply_fn({"params": p}, d))

            loss = self.criterion(
                logits=energy, targets=total_energy
            )  # FIXME: divide by n_atoms
            return loss, energy

        grad_fn = grad(loss_fn, has_aux=True)
        grads, energy = grad_fn(tuple(s.params for s in state))

        state = tuple(s.apply_gradients(grads=g) for s, g in zip(state, grads))
        metrics = mse_loss(
            logits=energy, targets=total_energy
        )  # FIXME: divide by n_atoms

        return state, metrics

    @partial(jit, static_argnums=(0, 1))  # FIXME
    def train_step_force(
        self,
        sargs: Tuple,
        state: Tuple[TrainState],
        batch: Tuple[Tuple, Tuple[jnp.ndarray]],
    ) -> Tuple:

        xbatch, target_forces = batch

        def loss_fn(
            params: Tuple[frozendict],
        ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray]]:
            """
            loss function
            """

            def energy_fn(xs: Tuple[jnp.array]) -> jnp.ndarray:
                """
                A helper function that allows to calculate the total energy gradient
                respect to the atom positions for each element.
                """
                energy = jnp.array(0.0)
                for s, p, inputs, static_inputs, x in zip(
                    state, params, xbatch, sargs, xs
                ):
                    _, position, atype, lattice, emap = inputs
                    asf, scaler, dtype = static_inputs

                    dsc = _calculate_descriptor(
                        asf, x, position, atype, lattice, dtype, emap
                    )
                    scaled_dsc = scaler(dsc)
                    energy += jnp.sum(s.apply_fn({"params": p}, scaled_dsc))

                return energy

            # Get positions for each element
            positions: Tuple[jnp.ndarray] = tuple(inputs[0] for inputs in xbatch)

            # Calculate forces for each element based on gradient of the total energy
            grad_energy_fn = grad(energy_fn)
            grad_energies: Tuple[jnp.ndarray] = grad_energy_fn(positions)
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

        # TODO: optimize (mask?), improve design
        def calculate_asf(structure) -> Tuple[jnp.ndarray]:
            def extract_element_asf():
                for element in self.potential.elements:
                    aid = structure.select(element)
                    x = self.potential.descriptor[element](structure, aid=aid)
                    x = self.potential.scaler[element](x)
                    yield x

            return tuple(asf for asf in extract_element_asf())

        def prepare_inputs(structure) -> Tuple[Tuple]:
            def extract_inputs():
                for element in self.potential.elements:
                    aid = structure.select(element)
                    yield (
                        structure.position[aid],
                        structure.position,
                        structure.atype,
                        structure.box.lattice,
                        structure.element_map.element_to_atype,
                    )

            return tuple(inputs for inputs in extract_inputs())

        def prepare_static_inputs(structure):
            def extract_static_inputs():
                for element in self.potential.elements:
                    yield (
                        self.potential.descriptor[element],
                        self.potential.scaler[element],
                        structure.dtype,
                    )

            return tuple(inputs for inputs in extract_static_inputs())

        def prepare_forces(structure) -> Tuple[jnp.ndarray]:
            def extract_force():
                for element in self.potential.elements:
                    aid = structure.select(element)
                    yield structure.force[aid]

            return tuple(force for force in extract_force())

        history = defaultdict(list)
        state = self.init_train_state()
        for epoch in range(50):

            for structure in dataset:

                batch = calculate_asf(structure), structure.total_energy
                state, metrics = self.train_step_energy(state, batch)

                sargs = prepare_static_inputs(structure)
                batch = prepare_inputs(structure), prepare_forces(structure)
                state, metrics = self.train_step_force(sargs, state, batch)

            if epoch % 1 == 0:
                print(f"Epoch {epoch}, loss={metrics}")
            history["epoch"].append(epoch)
            history["metrics_train"].append(metrics)

            # Update model params
            for element, train_state in zip(self.potential.elements, state):
                self.potential.model_params[element] = train_state.params

        return history

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\npotential={self.potential}, \noptimizer={self.optimizer}, "
            f"\ncriterion={self.criterion}, \nerror_metric={self.error_metric})"
        )
