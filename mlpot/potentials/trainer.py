from ..logger import logger
from ..potentials.base import Potential
from ..datasets.base import StructureDataset
from ..utils.gradient import gradient
from ..utils.attribute import set_as_attribute
from ..base import _Base
from .metrics import ErrorMetric, MSE
from collections import defaultdict
from typing import Dict
from torch.utils.data import DataLoader as TorchDataLoader
from torch import nn
import numpy as np
import torch


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

    def __init__(
        self,
        potential: Potential,
        optimizer: torch.optim.Optimizer,
        criterion: nn.MSELoss = nn.MSELoss(),
        error_metric: ErrorMetric = MSE(),
        save_best_model: bool = True,
        force_weight: float = 1.0,
        atom_energy: Dict[str, float] = None,
        **kwargs,
    ) -> None:
        """
        Initialize trainer.
        """
        self.potential = potential
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_best_model = save_best_model
        self.error_metric = error_metric
        self.force_weight = force_weight
        self.atom_energy = atom_energy
        set_as_attribute(self, kwargs)

        super().__init__()

    def fit_one_epoch(
        self,
        loader: TorchDataLoader,
        epoch_index: int = None,
        validation_mode: bool = False,
        history: Dict = None,
    ) -> Dict[str, float]:

        if validation_mode:
            self.potential.eval()
            prefix = "valid"
        else:
            self.potential.train()
            prefix = "train"

        nbatch: int = 0
        state: Dict[str, float] = defaultdict(float)
        # Loop over training structures
        for batch in loader:

            # Reset optimizer
            if not validation_mode:
                self.optimizer.zero_grad(set_to_none=True)

            # TODO: what if batch size > 1
            # TODO: spawn process
            structure = batch[0]
            # ---
            structure.set_cutoff_radius(self.potential.r_cutoff)
            if self.atom_energy:
                structure.remove_energy_offset(self.atom_energy)

            # Predict energy and force
            energy = self.potential(structure)  # total energy
            force = -gradient(energy, structure.position)

            # TODO: further investigation of possible input arguments to optimize grad calculations
            # torch._C._debug_only_display_vmap_fallback_warnings(True)
            # force = grad(energy, [structure.position],
            #             #grad_outputs = torch.ones_like(structure.position),
            #             # is_grads_batched = True,
            #             # create_graph = True,
            #             retain_graph=True,
            #   )[0]

            # Energy and force losses
            eng_loss = self.criterion(
                energy / structure.natoms, structure.total_energy / structure.natoms
            )  # TODO: optimize division
            frc_loss = self.criterion(force, structure.force)
            loss = eng_loss + self.force_weight * frc_loss

            # Error metrics
            eng_error = self.error_metric(
                energy, structure.total_energy, structure.natoms
            )
            frc_error = self.error_metric(force, structure.force)

            # ---
            if self.atom_energy:
                structure.add_energy_offset(self.atom_energy)

            # Update weights
            if not validation_mode and (epoch_index is None or epoch_index > 0):
                loss.backward(retain_graph=True)
                self.optimizer.step()

            # Accumulate energy and force loss and error values
            state[f"{prefix}_energy_loss"] += eng_loss.item()
            state[f"{prefix}_force_loss"] += frc_loss.item()
            state[f"{prefix}_loss"] += loss.item()
            state[f"{prefix}_energy_error"] += eng_error.item()
            state[f"{prefix}_force_error"] += frc_error.item()

            # Increment number of batches
            nbatch += 1

            if not validation_mode:
                logger.print(
                    "Training     "
                    f"loss: {state['train_loss']/nbatch :<12.8E}, "
                    f"energy [{self.error_metric.__class__.__name__}]: {state['train_energy_error']/nbatch :<12.8E}, "
                    f"force [{self.error_metric.__class__.__name__}]: {state['train_force_error']/nbatch :<12.8E}",
                    end="\r",
                )

        if validation_mode:
            logger.print(
                "Validation   "
                f"loss: {state['valid_loss'] :<12.8E}, "
                f"energy [{self.error_metric.__class__.__name__}]: {state['valid_energy_error']/nbatch :<12.8E}, "
                f"force [{self.error_metric.__class__.__name__}]: {state['valid_force_error']/nbatch :<12.8E}"
            )
        logger.print()

        # Mean values of loss and errors.
        for item in state:
            state[item] /= nbatch

        if history is not None:
            for item, value in state.items():
                history[item].append(value)

        return state

    def fit(self, dataset: StructureDataset, **kwargs) -> Dict:
        """
        Fit models.
        """
        # TODO: more arguments to have better control on training
        epochs = kwargs.get("epochs", 1)
        # TODO: add validation split from potential settings
        validation_split = kwargs.get("validation_split", None)
        validation_dataset = kwargs.get("validation_dataset", None)

        # Prepare structure dataset and loader for training elemental models
        # dataset_ = dataset.copy() # because of having new r_cutoff specific to the potential, no structure data will be copied
        # dataset_.transform = ToStructure(r_cutoff=self.potential.r_cutoff)

        # TODO: further optimization using the existing parameters in TorchDataloader
        # workers, pinned memory, etc.
        params = {
            "batch_size": 1,
            # "shuffle": True,
            # "num_workers": 4,
            # "prefetch_factor": 3,
            # "pin_memory": True,
            # "persistent_workers": True,
            "collate_fn": lambda batch: batch,
        }

        if validation_dataset:
            # Setting loaders
            train_loader = TorchDataLoader(dataset, shuffle=True, **params)
            valid_loader = TorchDataLoader(validation_dataset, shuffle=False, **params)
            # Logging
            logger.debug("Using separate training and validation datasets")
            logger.print(f"Number of structures (training)  : {len(dataset)}")
            logger.print(
                f"Number of structures (validation): {len(validation_dataset)}"
            )
            logger.print()

        elif validation_split:
            nsamples = len(dataset)
            split = int(np.floor(validation_split * nsamples))
            train_dataset, valid_dataset = torch.utils.data.random_split(
                dataset, lengths=[nsamples - split, split]
            )
            # Setting loaders
            train_loader = TorchDataLoader(train_dataset, shuffle=True, **params)
            valid_loader = TorchDataLoader(valid_dataset, shuffle=False, **params)
            # Logging
            logger.debug("Splitting dataset into training and validation subsets")
            logger.print(
                f"Number of structures (training)  : {nsamples - split} of {nsamples}"
            )
            logger.print(
                f"Number of structures (validation): {split} ({validation_split:0.2%} split)"
            )
            logger.print()

        else:
            train_loader = TorchDataLoader(dataset, shuffle=True, **params)
            valid_loader = None

        logger.debug("Fitting energy models")
        history = defaultdict(list)

        for epoch in range(epochs + 1):
            logger.print(f"[Epoch {epoch}/{epochs}]")

            # Train model for one epoch
            self.fit_one_epoch(train_loader, epoch, history=history)

            # Evaluate model on validation data
            if valid_loader is not None:
                self.fit_one_epoch(
                    valid_loader, epoch, history=history, validation_mode=True
                )

        # TODO: save the best model
        if self.save_best_model:
            self.potential.save_model()

        return history

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\npotential={self.potential}, \noptimizer={self.optimizer}, "
            f"\ncriterion={self.criterion}, \nerror_metric={self.error_metric})"
        )