import jax.numpy as jnp
import optax
import random
from collections import defaultdict
from typing import Dict, Callable, Tuple
from flax.training.train_state import TrainState
from frozendict import frozendict
from functools import partial
from jax import jit, value_and_grad
from mlpot.logger import logger
from mlpot.potentials.base import Potential
from mlpot.datasets.base import StructureDataset
from mlpot.base import _Base
from mlpot.potentials.loss import mse_loss
from mlpot.potentials.metrics import ErrorMetric
from mlpot.potentials.metrics import init_error_metric
from mlpot.potentials._energy import _energy_fn, _compute_forces
from mlpot.types import Array


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

    def init_train_state(self) -> Dict[str, TrainState]:
        """
        Initialize train state for each element.
        """

        def generate_train_state():
            for element in self.potential.elements:
                yield element, TrainState.create(
                    apply_fn=self.potential.model[element].apply,
                    params=self.potential.model_params[element],
                    tx=self.optimizer,  # [element]?
                )

        return {element: state for element, state in generate_train_state()}

    def fit(self, dataset: StructureDataset, **kwargs):

        static_args = self.potential.get_static_args()
        states: Dict[str, TrainState] = self.init_train_state()
        history = defaultdict(list)

        for epoch in range(20):

            # steps = len(dataset)
            for structure in random.choices(dataset, k=len(dataset)):

                xbatch = structure.get_inputs()
                ybatch = (
                    structure.total_energy,
                    structure.get_forces(),
                )

                # batch = xbatch, structure.total_energy
                # states, metrics_energy = self.train_step_energy(sargs, states, batch)

                batch = xbatch, ybatch
                states, metrics = self.train_step(static_args, states, batch)

            if epoch % 1 == 0:
                print(
                    f"Epoch {epoch}"
                    f", loss:{float(metrics['loss']): 0.7f}"
                    # f", loss[force]:{metrics_force['loss']: 0.7f}"
                )
            history["epoch"].append(epoch)
            history["loss"].append(metrics["loss"])

            for element, train_state in states.items():
                self.potential.model_params[element] = train_state.params

        return history

    @partial(jit, static_argnums=(0, 1))  # FIXME
    def train_step(
        self,
        static_args: Dict,
        states: Dict[str, TrainState],
        batch: Tuple,
    ) -> Tuple[Dict[str, TrainState], Dict]:

        xbatch, (true_energy, true_forces) = batch
        positions = {element: input.position_aid for element, input in xbatch.items()}
        n_atoms: int = sum(array.shape[0] for array in positions.values())
        elements = true_forces.keys()

        def loss_fn(
            params: Dict[str, frozendict],
        ) -> Tuple[Array, Tuple]:
            """
            Loss function
            """
            loss = jnp.array(0.0)

            # Energy
            energy = _energy_fn(static_args, positions, params, xbatch)
            loss_eng = self.criterion(logits=energy, targets=true_energy) / n_atoms
            loss += loss_eng

            # Force
            forces = _compute_forces(static_args, positions, params, xbatch)
            # TODO: define loss weights for each element
            loss_frc = jnp.array(0.0)
            for element in elements:
                loss_frc += self.criterion(
                    logits=forces[element],
                    targets=true_forces[element],
                )
            loss_frc /= len(forces)
            # TODO: add coefficient
            loss += loss_frc / n_atoms

            return loss, (energy, forces)

        value_and_grad_fn = value_and_grad(loss_fn, has_aux=True)
        (loss, (energy, forces)), grads = value_and_grad_fn(
            {element: state.params for element, state in states.items()}
        )
        states = {
            element: states[element].apply_gradients(grads=grads[element])
            for element in elements
        }

        # TODO: add more metrics for force
        metrics = {
            "loss": loss,
        }
        return states, metrics

    # @partial(jit, static_argnums=(0, 1))  # FIXME
    # def train_step_energy(
    #     self,
    #     sargs: Tuple,
    #     states: Tuple[TrainState],
    #     batch: Tuple[Tuple, Tuple[Array]],
    # ) -> Tuple[Tuple[TrainState], Dict]:

    #     xbatch, target_energy = batch

    #     positions: Tuple[Array] = tuple(inputs[0] for inputs in xbatch)
    #     n_atoms: int = sum(pos.shape[0] for pos in positions)

    #     def loss_fn(params: Tuple[frozendict]) -> Tuple[Array, Array]:
    #         """
    #         Loss function based on energy.
    #         """
    #         energy = _energy_fn(sargs, positions, params, xbatch)
    #         loss = self.criterion(logits=energy, targets=target_energy) / n_atoms

    #         return loss, energy

    #     value_and_grad_fn = value_and_grad(loss_fn, has_aux=True)
    #     (loss, energy), grads = value_and_grad_fn(tuple(s.params for s in states))
    #     states = tuple(s.apply_gradients(grads=g) for s, g in zip(states, grads))

    #     metrics = {
    #         "loss": loss,
    #         "error": target_energy - energy,
    #     }
    #     return states, metrics

    # def eval_step(self, batch) -> None:
    #     pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\npotential={self.potential}, \noptimizer={self.optimizer}, "
            f"\ncriterion={self.criterion}, \nerror_metric={self.error_metric})"
        )
