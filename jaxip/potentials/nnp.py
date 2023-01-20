from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import jax.numpy as jnp
from frozendict import frozendict
from jax import random
from this import d
from tqdm import tqdm

from jaxip.datasets.runner import RunnerStructureDataset
from jaxip.descriptors.acsf.acsf import ACSF
from jaxip.descriptors.scaler import DescriptorScaler
from jaxip.logger import logger
from jaxip.models.nn import NeuralNetworkModel
from jaxip.potentials._energy import _compute_energy
from jaxip.potentials._force import _compute_force
from jaxip.potentials.atomic_potential import AtomicPotential
from jaxip.potentials.settings import NeuralNetworkPotentialSettings
from jaxip.potentials.trainer import NeuralNetworkPotentialTrainer
from jaxip.structure import Structure
from jaxip.structure.element import ElementMap
from jaxip.types import Array, Element


@dataclass
class NeuralNetworkPotential:
    """
    High-dimensional neural network potential (HDNNP) - second generation.

    It contains all the required descriptors, scalers, and neural networks for each element,
    and a trainer to fit the potential using reference structure data.
    """

    potential_file: Path = Path("input.data")
    atomic_potential: Dict[Element, AtomicPotential] = field(default_factory=dict)
    model_params: Dict[Element, frozendict] = field(default_factory=dict, repr=False)
    trainer: Optional[NeuralNetworkPotentialTrainer] = field(default=None, repr=False)
    settings: Optional[NeuralNetworkPotentialSettings] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Post initialize potential parameters."""
        self.potential_file = Path(self.potential_file)

        if self.settings is None:
            self.settings = NeuralNetworkPotentialSettings(filename=self.potential_file)

        if not self.atomic_potential:
            self.init_atomic_potential()

        if len(self.model_params) == 0:
            self.init_model_params()

        if self.trainer is None:
            logger.debug("[Setting trainer]")
            self.trainer = NeuralNetworkPotentialTrainer(potential=self)

    def init_atomic_potential(self) -> None:
        """
        Initialize atomic potential for each element.

        This method can be override in case that different atomic potential is used.
        """
        descriptor: Dict[Element, ACSF] = self.settings.get_descriptor()
        scaler: Dict[Element, DescriptorScaler] = self.settings.get_scaler()
        model: Dict[Element, NeuralNetworkModel] = self.settings.get_model()
        for element in self.elements:
            self.atomic_potential[element] = AtomicPotential(
                descriptor=descriptor[element],
                scaler=scaler[element],
                model=model[element],
            )

    def init_model_params(self) -> None:
        """
        Initialize neural network model parameters for each element
        (e.g. neural network kernel and bias values).

        This method be used to initialize model params of the potential with a different random seed.
        """
        random_keys = random.split(
            random.PRNGKey(self.settings["random_seed"]),
            self.settings["number_of_elements"],
        )
        for i, element in enumerate(self.settings["elements"]):
            self.model_params[element] = atomic_potential[element].model.init(  # type: ignore
                random_keys[i],
                jnp.ones((1, self.atomic_potential[element].model_input_size)),
            )[
                "params"
            ]

    # ------------------------------------------------------------------------

    def __call__(self, structure: Structure) -> Array:
        """
        Compute the total energy.

        :param structure: Structure
        :return: total energy
        """
        return _compute_energy(
            frozendict(self.atomic_potential),  # must be hashable
            structure.get_positions(),
            self.model_params,
            structure.get_inputs(),
        )

    def compute_force(self, structure: Structure) -> Dict[Element, Array]:
        """
        Compute force components.

        :param structure: input structure
        :return: per-atom force components
        """
        forces: Dict[Element, Array] = _compute_force(
            frozendict(self.atomic_potential),  # must be hashable
            structure.get_positions(),
            self.model_params,
            structure.get_inputs(),
        )
        return forces

    # ------------------------------------------------------------------------

    # @Profiler.profile
    def fit_scaler(self, dataset: RunnerStructureDataset, **kwargs) -> None:
        """
        Fit scaler parameters for each element using the input structure data.
        No gradient history is required here.
        """
        save_scaler: bool = kwargs.get("save_scaler", True)

        # loader = TorchDataLoader(dataset, collate_fn=lambda batch: batch)
        print("Fitting descriptor scaler...")
        for structure in tqdm(dataset):
            for element in structure.elements:
                x = self.atomic_potential[element].descriptor(structure)
                self.atomic_potential[element].scaler.fit(x)
        print("Done.\n")

        if save_scaler:
            self.save_scaler()

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
        return self.trainer.fit(dataset, **kwargs)  # type: ignore

    def fit(self) -> None:
        """
        This method provides a user-friendly interface to fit both descriptor and model in one step.
        """
        pass

    # ------------------------------------------------------------------------

    def save_scaler(self) -> None:
        """
        This method saves scaler parameters for each element into separate files.
        """
        # Save scaler parameters for each element separately
        for element in self.elements:
            atomic_number: int = ElementMap.get_atomic_number(element)
            scaler_file = Path(
                self.potential_file.parent,
                self.settings["scaler_save_naming_format"].format(atomic_number),
            )
            logger.info(
                f"Saving scaler parameters for element ({element}): {scaler_file.name}"
            )
            self.atomic_potential[element].scaler.save(scaler_file)

    def save_model(self) -> None:
        """
        Save model weights separately for all elements.
        """
        for element in self.elements:
            logger.debug(f"Saving model weights for element: {element}")
            atomic_number = ElementMap.get_atomic_number(element)
            model_file = Path(
                self.potential_file.parent,
                self.settings["model_save_naming_format"].format(atomic_number),
            )
            self.atomic_potential[element].model.save(model_file)

    # @Profiler.profile
    def load_scaler(self) -> None:
        """
        This method loads scaler parameters of each element from separate files.
        This save computational time as the would be no need to fit the scalers each time.
        """
        # Load scaler parameters for each element separately
        for element in self.elements:
            atomic_number: int = ElementMap.get_atomic_number(element)
            scaler_file = Path(
                self.potfile.parent,
                self.settings["scaler_save_naming_format"].format(atomic_number),
            )
            logger.debug(
                f"Loading scaler parameters for element {element}: {scaler_file.name}"
            )
            self.atomic_potential[element].scaler.load(scaler_file)

    # @Profiler.profile
    def load_model(self) -> None:
        """
        Load model weights separately for all elements.
        """
        for element in self.elements:
            logger.debug(f"Loading model weights for element: {element}")
            atomic_number: int = ElementMap.get_atomic_number(element)
            model_file = Path(
                self.potential_file.parent,
                self.settings["model_save_naming_format"].format(atomic_number),
            )
            self.atomic_potential[element].model.load(model_file)

    def set_extrapolation_warnings(self, threshold: Optional[int] = None) -> None:
        """
        shows warning whenever a descriptor value is out of bounds defined by
        minimum/maximum values in the scaler.

        set_extrapolation_warnings(None) will disable it.

        :param threshold: maximum number of warnings
        :type threshold: int
        """
        logger.info(f"Setting extrapolation warning: {threshold}")
        for pot in self.atomic_potential.values():
            pot.scaler.set_max_number_of_warnings(threshold)

    # def __repr__(self) -> str:
    #     return f"{self.__class__.__name__}(potfile='{self.potfile.name}')"

    @property
    def extrapolation_warnings(self) -> Dict[Element, int]:
        return {
            element: pot.scaler.number_of_warnings
            for element, pot in self.atomic_potential.items()
        }

    @property
    def elements(self) -> List[Element]:
        """Return elements."""
        return self.settings["elements"]

    @property
    def num_elements(self) -> int:
        """Return number of elements."""
        return len(self.elements)

    @property
    def r_cutoff(self) -> float:
        """Return the maximum cutoff radius found between all descriptors."""
        return max([pot.descriptor.r_cutoff for pot in self.atomic_potential.values()])

    @property
    def descriptor(self) -> Dict:
        """Return descriptor for each element."""
        return {elem: pot.descriptor for elem, pot in self.atomic_potential.items()}

    @property
    def scaler(self) -> Dict:
        """Return scaler for each element."""
        return {elem: pot.scaler for elem, pot in self.atomic_potential.items()}

    @property
    def model(self) -> Dict:
        """Return model for each element."""
        return {elem: pot.model for elem, pot in self.atomic_potential.items()}
