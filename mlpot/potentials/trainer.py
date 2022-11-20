from ..logger import logger
from ..potentials.base import Potential
from ..datasets.base import StructureDataset
from ..utils.gradient import gradient
from ..utils.attribute import set_as_attribute
from ..base import _Base
from .loss import mse_loss
from .metrics import ErrorMetric
from .metrics import init_error_metric
from collections import defaultdict
from typing import Dict, Callable, Tuple
from flax.training.train_state import TrainState
import numpy as np
import optax

import jax.numpy as jnp


Tensor = jnp.ndarray


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

        # self.force_weight: float = self.potential.settings["force_weight"]
        # self.atom_energy: Dict[str, float] = self.potential.settings["atom_energy"]

        self.optimizer = self.init_optimizer()
        self.train_state = self.init_train_state()

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

    def init_train_state(self) -> Dict:

        train_state: Dict[str, TrainState] = dict()
        for element in self.potential.elements:
            train_state[element] = TrainState.create(
                apply_fn=self.potential.model[element].apply,
                params=self.potential.model_params[element],
                tx=self.optimizer,  # [element]?
            )

        return train_state

    def eval_step(self, batch):
        pass

    # def train_step(
    #     self,
    #     state: Dict[str, TrainState],
    #     batch,
    #     # epoch_index: int = None,
    #     # validation_mode: bool = False,
    #     # history: Dict = None,
    # ) -> Dict[str, float]:

    #     def loss_fn(state):
    #         logits = state.apply_fn({'params': params}, x)
    #         self.criterion()

    #     if validation_mode:
    #         self.potential.eval()
    #         prefix = "valid"
    #     else:
    #         self.potential.train()
    #         prefix = "train"

    #     nbatch: int = 0
    #     state: Dict[str, float] = defaultdict(float)
    #     error_metric_name = self.error_metric__class__.__name__

    #     # Loop over training structures
    #     for batch in loader:

    #         # Reset optimizer
    #         if not validation_mode:
    #             self.optimizer.zero_grad(set_to_none=True)

    #         # TODO: what if batch size > 1
    #         # TODO: spawn process
    #         structure = batch[0]
    #         # ---
    #         structure.set_cutoff_radius(self.potential.r_cutoff)
    #         if self.atom_energy:
    #             structure.remove_energy_offset(self.atom_energy)

    #         # Predict energy and force
    #         energy = self.potential(structure)  # total energy
    #         force = -gradient(energy, structure.position)

    #         # TODO: further investigation of possible input arguments to optimize grad calculations
    #         # torch._C._debug_only_display_vmap_fallback_warnings(True)
    #         # force = grad(energy, [structure.position],
    #         #             #grad_outputs = torch.ones_like(structure.position),
    #         #             # is_grads_batched = True,
    #         #             # create_graph = True,
    #         #             retain_graph=True,
    #         #   )[0]

    #         # Energy and force losses
    #         eng_loss = self.criterion(
    #             energy / structure.n_atoms, structure.total_energy / structure.n_atoms
    #         )  # TODO: optimize division
    #         frc_loss = self.criterion(force, structure.force)
    #         loss = eng_loss + self.force_weight * frc_loss

    #         # Error metrics
    #         eng_error = self.error_metric(
    #             energy, structure.total_energy, structure.n_atoms
    #         )
    #         frc_error = self.error_metric(force, structure.force)

    #         # ---
    #         if self.atom_energy:
    #             structure.add_energy_offset(self.atom_energy)

    #         # Update weights
    #         if not validation_mode and (epoch_index is None or epoch_index > 0):
    #             loss.backward(retain_graph=True)
    #             self.optimizer.step()

    #         # Accumulate energy and force loss and error values
    #         state[f"{prefix}_energy_loss"] += eng_loss.item()
    #         state[f"{prefix}_force_loss"] += frc_loss.item()
    #         state[f"{prefix}_loss"] += loss.item()
    #         state[f"{prefix}_energy_error"] += eng_error.item()
    #         state[f"{prefix}_force_error"] += frc_error.item()

    #         # Increment number of batches
    #         nbatch += 1

    #         if not validation_mode:
    #             logger.print(
    #                 "Training     "
    #                 f"loss: {state['train_loss']/nbatch :<12.8E}, "
    #                 f"energy [{error_metric_name}]: {state['train_energy_error']/nbatch :<12.8E}, "
    #                 f"force [{error_metric_name}]: {state['train_force_error']/nbatch :<12.8E}",
    #                 end="\r",
    #             )

    #     if validation_mode:
    #         logger.print(
    #             "Validation   "
    #             f"loss: {state['valid_loss'] :<12.8E}, "
    #             f"energy [{error_metric_name}]: {state['valid_energy_error']/nbatch :<12.8E}, "
    #             f"force [{error_metric_name}]: {state['valid_force_error']/nbatch :<12.8E}"
    #         )
    #     logger.print()

    #     # Mean values of loss and errors.
    #     for item in state:
    #         state[item] /= nbatch

    #     if history is not None:
    #         for item, value in state.items():
    #             history[item].append(value)

    #     return state

    # def fit(self, dataset: StructureDataset, **kwargs) -> Dict:
    #     """
    #     Fit models.
    #     """
    #     # TODO: more arguments to have better control on training
    #     epochs = kwargs.get("epochs", 1)
    #     # TODO: add validation split from potential settings
    #     validation_split = kwargs.get("validation_split", None)
    #     validation_dataset = kwargs.get("validation_dataset", None)

    #     # Prepare structure dataset and loader for training elemental models
    #     # dataset_ = dataset.copy() # because of having new r_cutoff specific to the potential, no structure data will be copied
    #     # dataset_.transform = ToStructure(r_cutoff=self.potential.r_cutoff)

    #     # TODO: further optimization using the existing parameters in TorchDataloader
    #     # workers, pinned memory, etc.
    #     params = {
    #         "batch_size": 1,
    #         # "shuffle": True,
    #         # "num_workers": 4,
    #         # "prefetch_factor": 3,
    #         # "pin_memory": True,
    #         # "persistent_workers": True,
    #         "collate_fn": lambda batch: batch,
    #     }

    #     if validation_dataset:
    #         # Setting loaders
    #         train_loader = TorchDataLoader(dataset, shuffle=True, **params)
    #         valid_loader = TorchDataLoader(validation_dataset, shuffle=False, **params)
    #         # Logging
    #         logger.debug("Using separate training and validation datasets")
    #         logger.print(f"Number of structures (training)  : {len(dataset)}")
    #         logger.print(
    #             f"Number of structures (validation): {len(validation_dataset)}"
    #         )
    #         logger.print()

    #     elif validation_split:
    #         nsamples = len(dataset)
    #         split = int(np.floor(validation_split * nsamples))
    #         train_dataset, valid_dataset = torch.utils.data.random_split(
    #             dataset, lengths=[nsamples - split, split]
    #         )
    #         # Setting loaders
    #         train_loader = TorchDataLoader(train_dataset, shuffle=True, **params)
    #         valid_loader = TorchDataLoader(valid_dataset, shuffle=False, **params)
    #         # Logging
    #         logger.debug("Splitting dataset into training and validation subsets")
    #         logger.print(
    #             f"Number of structures (training)  : {nsamples - split} of {nsamples}"
    #         )
    #         logger.print(
    #             f"Number of structures (validation): {split} ({validation_split:0.2%} split)"
    #         )
    #         logger.print()

    #     else:
    #         train_loader = TorchDataLoader(dataset, shuffle=True, **params)
    #         valid_loader = None

    #     logger.debug("Fitting energy models")
    #     history = defaultdict(list)

    #     for epoch in range(epochs + 1):
    #         logger.print(f"[Epoch {epoch}/{epochs}]")

    #         # Train model for one epoch
    #         self.fit_one_epoch(train_loader, epoch, history=history)

    #         # Evaluate model on validation data
    #         if valid_loader is not None:
    #             self.fit_one_epoch(
    #                 valid_loader, epoch, history=history, validation_mode=True
    #             )

    #     # TODO: save the best model
    #     if self.save_best_model:
    #         self.potential.save_model()

    #     return history

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\npotential={self.potential}, \noptimizer={self.optimizer}, "
            f"\ncriterion={self.criterion}, \nerror_metric={self.error_metric})"
        )
