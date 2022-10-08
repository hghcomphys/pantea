from torchip.descriptors.base import Descriptor
from ..logger import logger
from ..structure import Structure
from ..datasets.runner import RunnerStructureDataset
from ..descriptors.asf.asf import ASF
from ..descriptors.asf.cutoff import CutoffFunction
from ..descriptors.scaler import DescriptorScaler
from ..descriptors.asf.radial import G1, G2
from ..descriptors.asf.angular import G3, G9
from ..models.nn import NeuralNetworkModel
from ..utils.batch import create_batch
from ..utils.profiler import Profiler
from ..structure.element import ElementMap
from ..config import dtype, device
from .base import Potential
from .settings import NeuralNetworkPotentialSettings
from .trainer import NeuralNetworkPotentialTrainer
from .metric import create_error_metric
from typing import List, Dict
from torch.utils.data import DataLoader as TorchDataLoader
from pathlib import Path
from torch import Tensor
from torch import nn
import torch


class NeuralNetworkPotential(Potential):
    """
    A suitcase class of high-dimensional neural network potential (HDNNP).
    It contains all the required descriptors, scalers, neural networks, and a trainer to fit the potential using
    provided structure data and a potential setting file.
    """

    # TODO: split structures from the potential model
    # TODO: implement structure dumper/writer
    # TODO: split structure from the potential model (in design)

    # Saving formats
    _scaler_save_format: str = "scaling.{:03d}.data"
    _model_save_format: str = "weights.{:03d}.zip"

    def __init__(self, potfile: Path) -> None:
        """
        Initialize a HDNNP potential instance by reading the potential file and
        creating descriptors, scalers, models, and trainer.

        Args:
            potfile (Path): A file path to potential file.
        """
        self.potfile = Path(potfile)
        self.settings = NeuralNetworkPotentialSettings()
        super().__init__()

        self.settings.read(self.potfile)

        self.descriptor: Dict[str, Descriptor] = self._init_descriptor()
        self.scaler: Dict[str, DescriptorScaler] = self._init_scaler()
        self.model: Dict[str, NeuralNetworkModel] = self._init_model()
        self.trainer: NeuralNetworkPotentialTrainer = self._init_trainer()

    def _init_descriptor(self) -> Dict[str, Descriptor]:
        """
        Initialize a **descriptor** for each element and add the relevant radial and angular
        symmetry functions from the potential settings.
        """
        # TODO: add logging
        logger.debug(f"[Setting descriptors]")
        descriptor = {}

        # Elements
        logger.info(f"Number of elements: {len(self.settings['elements'])}")
        for element in self.settings["elements"]:
            logger.info(f"Element: {element} ({ElementMap.get_atomic_number(element)})")

        # Instantiate ASF for each element
        for element in self.settings["elements"]:
            descriptor[element] = ASF(element)

        # Add symmetry functions
        logger.debug(
            f"Registering symmetry functions (radial and angular)"
        )  # TODO: move logging inside .add() method

        for cfg in self.settings["symfunction_short"]:
            if cfg[1] == 1:
                descriptor[cfg[0]].register(
                    symmetry_function=G1(
                        CutoffFunction(
                            r_cutoff=cfg[5], cutoff_type=self.settings["cutoff_type"]
                        )
                    ),
                    neighbor_element1=cfg[2],
                )
            elif cfg[1] == 2:
                descriptor[cfg[0]].register(
                    symmetry_function=G2(
                        CutoffFunction(
                            r_cutoff=cfg[5], cutoff_type=self.settings["cutoff_type"]
                        ),
                        eta=cfg[3],
                        r_shift=cfg[4],
                    ),
                    neighbor_element1=cfg[2],
                )
            elif cfg[1] == 3:
                descriptor[cfg[0]].register(
                    symmetry_function=G3(
                        CutoffFunction(
                            r_cutoff=cfg[7], cutoff_type=self.settings["cutoff_type"]
                        ),
                        eta=cfg[4],
                        zeta=cfg[6],
                        lambda0=cfg[5],
                        r_shift=0.0,
                    ),  # TODO: add r_shift!
                    neighbor_element1=cfg[2],
                    neighbor_element2=cfg[3],
                )
            elif cfg[1] == 9:
                descriptor[cfg[0]].register(
                    symmetry_function=G9(
                        CutoffFunction(
                            r_cutoff=cfg[7], cutoff_type=self.settings["cutoff_type"]
                        ),
                        eta=cfg[4],
                        zeta=cfg[6],
                        lambda0=cfg[5],
                        r_shift=0.0,
                    ),  # TODO: add r_shift!
                    neighbor_element1=cfg[2],
                    neighbor_element2=cfg[3],
                )

        return descriptor

    def _init_scaler(self) -> Dict[str, DescriptorScaler]:
        """
        Initialize a descriptor scaler for each element from the potential settings.
        """
        logger.debug(f"[Setting scalers]")
        scaler = dict()

        # Prepare scaler input argument if exist in settings
        scaler_kwargs = {
            first: self.settings[second]
            for first, second in {
                "scale_type": "scale_type",
                "scale_min": "scale_min_short",
                "scale_max": "scale_max_short",
            }.items()
            if second in self.settings.keywords
        }
        logger.debug(f"Scaler kwargs={scaler_kwargs}")

        # Assign an ASF scaler to each element
        for element in self.settings["elements"]:
            scaler[element] = DescriptorScaler(**scaler_kwargs)

        return scaler

    def _init_model(self) -> Dict[str, NeuralNetworkModel]:
        """
        Initialize a neural network for each element using a dictionary representation of potential settings.
        """
        logger.debug(f"[Setting models]")
        model = dict()

        # Instantiate neural network model for each element
        hidden_layers = zip(
            self.settings["global_nodes_short"],
            self.settings["global_activation_short"][:-1],
        )
        output_layer = (1, self.settings["global_activation_short"][-1])
        # TODO: what if we want to have a different model architecture for each element

        for element in self.settings["elements"]:
            logger.debug(f"Element: {element}")
            input_size = self.descriptor[element].n_descriptor
            model_kwargs = {
                "input_size": input_size,
                "hidden_layers": tuple([(n, t) for n, t in hidden_layers]),
                "output_layer": output_layer,
                "weights_range": (
                    self.settings["weights_min"],
                    self.settings["weights_max"],
                ),
            }
            model[element] = NeuralNetworkModel(**model_kwargs)
            model[element].to(device.DEVICE)

        return model

    def _init_trainer(self) -> NeuralNetworkPotentialTrainer:
        """
        This method initializes a trainer instance including optimizer, loss function, criterion, etc from the settings.
        The trainer is used later for fitting the energy models.
        """
        logger.debug(f"[Setting trainer]")
        trainer_kwargs = dict()

        # General trainer parameters
        trainer_kwargs["criterion"] = nn.MSELoss()
        trainer_kwargs["error_metric"] = create_error_metric(
            self.settings["main_error_metric"], reduction="mean"
        )
        trainer_kwargs["force_weight"] = self.settings["force_weight"]
        trainer_kwargs["atom_energy"] = self.settings["atom_energy"]

        if self.settings["updater_type"] == 0:  # Gradient Descent
            if self.settings["gradient_type"] == 1:  # Adam
                trainer_kwargs["learning_rate"] = self.settings[
                    "gradient_adam_eta"
                ]  # TODO: defining learning_rate?
                trainer_kwargs["optimizer_func"] = torch.optim.Adam
                trainer_kwargs["optimizer_func_kwargs"] = {
                    "lr": self.settings["gradient_adam_eta"],
                    "betas": (
                        self.settings["gradient_adam_beta1"],
                        self.settings["gradient_adam_beta2"],
                    ),
                    "eps": self.settings["gradient_adam_epsilon"],
                    "weight_decay": self.settings["gradient_weight_decay"],
                }
            elif self.settings["gradient_type"] == 0:  # Fixed Step
                logger.error(
                    "Gradient descent type fixed step is not implemented yet",
                    exception=NotImplementedError,
                )

        # Create trainer instance
        logger.debug(f"Trainer kwargs={trainer_kwargs}")
        return NeuralNetworkPotentialTrainer(self, **trainer_kwargs)

    # @Profiler.profile
    @torch.no_grad()
    def fit_scaler(self, dataset: RunnerStructureDataset, **kwargs) -> None:
        """
        Fit scaler parameters for each element using the input structure data.
        No gradient history is required here.

        Args:
            structures (RunnerStructureDataset): Structure dataset
        """
        # Set parameters
        save_scaler: bool = kwargs.get("save_scaler", True)
        batch_size: int = kwargs.get(
            "batch_size", 8
        )  # batch of atoms in each structure

        loader = TorchDataLoader(dataset, collate_fn=lambda batch: batch)

        logger.info("Fitting descriptor scalers")
        for index, batch in enumerate(loader):

            # TODO: spawn processes
            structure = batch[0]
            structure.set_cutoff_radius(self.r_cutoff)

            # For each element in the structure
            for element in structure.elements:
                aids = structure.select(element).detach()
                for aids_batch in create_batch(
                    aids, batch_size
                ):  # because of large memory usage
                    logger.debug(
                        f"Calculating descriptor for Structure {index} (element='{element}', aids={aids_batch})"
                    )
                    dsc_val = self.descriptor[element](structure, aids_batch)  # kernel
                    self.scaler[element].fit(dsc_val)
        logger.debug("Finished scaler fitting.")

        # Save scaler data into file
        if save_scaler:
            self.save_scaler()

    def save_scaler(self):
        """
        This method saves scaler parameters for each element into separate files.
        """
        # Save scaler parameters for each element separately
        for element in self.elements:
            atomic_number = ElementMap.get_atomic_number(element)
            scaler_fn = Path(
                self.potfile.parent, self._scaler_save_format.format(atomic_number)
            )
            logger.info(
                f"Saving scaler parameters for element ({element}): {scaler_fn.name}"
            )
            self.scaler[element].save(scaler_fn)

    # @Profiler.profile
    def load_scaler(self) -> None:
        """
        This method loads scaler parameters of each element from separate files.
        This save computational time as the would be no need to fit the scalers each time.
        """
        # Load scaler parameters for each element separately
        for element in self.elements:
            atomic_number = ElementMap.get_atomic_number(element)
            scaler_fn = Path(
                self.potfile.parent, self._scaler_save_format.format(atomic_number)
            )
            logger.debug(
                f"Loading scaler parameters for element {element}: {scaler_fn.name}"
            )
            self.scaler[element].load(scaler_fn)

    # @Profiler.profile
    def fit_model(self, dataset: RunnerStructureDataset, **kwargs) -> Dict:
        """
        Fit energy model for all elements using the input structure loader.
        """
        # TODO: avoid reading and calculating descriptor multiple times
        # TODO: descriptor element should be the same atom type as the aid
        # TODO: define a dataloader specific to energy and force data (shuffle, train & test split)
        # TODO: add validation output (MSE separate for force and energy)
        kwargs["validation_split"] = kwargs.get(
            "validation_split", self.settings["test_fraction"]
        )
        kwargs["epochs"] = kwargs.get("epochs", self.settings["epochs"])
        return self.trainer.fit(dataset, **kwargs)

    def save_model(self):
        """
        Save model weights separately for all elements.
        """
        for element in self.elements:
            logger.debug(f"Saving model weights for element: {element}")
            atomic_number = ElementMap.get_atomic_number(element)
            model_fn = Path(
                self.potfile.parent, self._model_save_format.format(atomic_number)
            )
            self.model[element].save(model_fn)

    # @Profiler.profile
    def load_model(self):
        """
        Load model weights separately for all elements.
        """
        for element in self.elements:
            logger.debug(f"Loading model weights for element: {element}")
            atomic_number = ElementMap.get_atomic_number(element)
            model_fn = Path(
                self.potfile.parent, self._model_save_format.format(atomic_number)
            )
            self.model[element].load(model_fn)

    def fit(self) -> None:
        """
        This method provides a user-friendly interface to fit both descriptor and model in one step.
        """
        pass

    def train(self, mode: bool = True) -> None:
        """
        Set pytorch models in training mode.
        This is because layers like dropout, batch normalization etc. behave differently on the train and test procedures.
        """
        for element in self.elements:
            self.model[element].train(mode)

    def eval(self) -> None:
        """
        Set pytorch models in evaluation mode.
        This is because layers like dropout, batch normalization etc. behave differently on the train and test procedures.
        """
        for element in self.elements:
            self.model[element].eval()

    def __call__(self, structure: Structure) -> Tensor:
        """
        Return the total energy of the input structure.

        :param structure: Structure
        :type structure: Structure
        :return: total energy
        :rtype: Tensor
        """
        structure.set_cutoff_radius(self.r_cutoff)

        # Set model in evaluation model
        self.eval()

        # Loop over elements
        energy = 0.0
        for element in self.elements:
            aids = structure.select(element).detach()
            x = self.descriptor[element](structure, aid=aids)
            x = self.scaler[element](x)
            x = self.model[element](x)
            # FIXME: float type neural network
            x = torch.sum(x, dim=0)
            energy = energy + x

        return energy

    @property
    def elements(self) -> List[str]:
        return self.settings["elements"]

    @property
    def r_cutoff(self) -> float:
        """
        Return the maximum cutoff radius found between all descriptors.
        """
        return max([dsc.r_cutoff for dsc in self.descriptor.values()])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(potfile='{self.potfile.name}')"
