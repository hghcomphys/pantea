from dataclasses import dataclass, field
from pathlib import Path
from turtle import st
from typing import Dict, List, Optional

import jax.numpy as jnp
from frozendict import frozendict
from jax import random
from tqdm import tqdm

from jaxip.datasets.runner import RunnerStructureDataset
from jaxip.descriptors.base import Descriptor
from jaxip.descriptors.scaler import DescriptorScaler
from jaxip.logger import logger
from jaxip.models.initializer import UniformInitializer
from jaxip.models.nn import NeuralNetworkModel
from jaxip.potentials._energy import _compute_force, _energy_fn
from jaxip.potentials.settings import NeuralNetworkPotentialSettings
from jaxip.potentials.trainer import NeuralNetworkPotentialTrainer
from jaxip.structure import Structure
from jaxip.structure.element import ElementMap
from jaxip.types import Array


@dataclass(frozen=True)
class AtomicPotential:
    """Atomic potential."""

    descriptor: Descriptor
    scaler: DescriptorScaler
    model: NeuralNetworkModel

    def apply(self, params: frozendict, structure: Structure) -> Array:
        x = self.descriptor(structure)
        x = self.scaler(x)
        x = self.model.apply({"params": params}, x)
        return x  # type: ignore


@dataclass
class NeuralNetworkPotential:
    """
    High-dimensional neural network potential (HDNNP) - second generation.

    It contains all the required descriptors, scalers, and neural networks for each element,
    and a trainer to fit the potential using reference structure data.
    """

    potfile: Path
    atomic_potential: Dict[str, AtomicPotential] = field(default_factory=dict)
    model_params: Dict[str, frozendict] = field(default_factory=dict)
    trainer: Optional[NeuralNetworkPotentialTrainer] = None

    def __post_init__(self) -> None:
        """Post initialize potential parameters."""
        # TODO: input settings externally (it conflicts with the potfile input)
        self.settings: NeuralNetworkPotentialSettings = NeuralNetworkPotentialSettings(
            filename=self.potfile
        )
        if len(self.atomic_potential) == 0:
            self._init_atomic_potential()
        if len(self.model_params) == 0:
            self._init_model_params()
        if self.trainer is None:
            logger.debug("[Setting trainer]")
            self.trainer = NeuralNetworkPotentialTrainer(potential=self)

    def _init_atomic_potential(self) -> None:
        """Initialize atomic potential for each element."""
        descriptor: Dict[str, Descriptor] = self.settings.get_descriptor()
        scaler: Dict[str, DescriptorScaler] = self.settings.get_scaler()
        model: Dict[str, NeuralNetworkModel] = self.settings.get_model()
        for element in self.elements:
            self.atomic_potential[element] = AtomicPotential(
                descriptor=descriptor[element],
                scaler=scaler[element],
                model=model[element],
            )

    def _init_model_params(self) -> None:
        """Initialize neural network model parameters for each element using model and descriptor parameters."""
        random_keys = random.split(random.PRNGKey(0), self.num_elements)
        for i, element in enumerate(self.elements):
            self.model_params[element] = self.atomic_potential[element].model.init(  # type: ignore
                random_keys[i],
                jnp.ones(
                    (1, self.atomic_potential[element].descriptor.num_descriptors)
                ),
            )[
                "params"
            ]

    # ------------------------------------------------------------------------

    def get_atomic_potential(self) -> frozendict:
        # return frozendict(
        #     {element: self.atomic_potential[element] for element in self.elements}
        # )
        return frozendict(self.atomic_potential)

    def __call__(self, structure: Structure) -> Array:
        """
        Return the total energy of the input structure.

        :param structure: Structure
        :type structure: Structure
        :return: total energy
        :rtype: Array
        """
        return _energy_fn(
            frozendict(self.atomic_potential),  # must be hashable
            structure.get_positions(),
            self.model_params,
            structure.get_inputs(),
        )

    def compute_force(self, structure: Structure) -> Dict[str, Array]:
        """Compute force components for all atoms in the input structure."""
        forces: Dict[str, Array] = _compute_force(
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
                self.potfile.parent,
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
                self.potfile.parent,
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
                self.potfile.parent,
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(potfile='{self.potfile.name}')"

    @property
    def extrapolation_warnings(self) -> Dict[str, int]:
        return {
            element: pot.scaler.number_of_warnings
            for element, pot in self.atomic_potential.items()
        }

    @property
    def elements(self) -> List[str]:
        return self.settings["elements"]

    @property
    def num_elements(self) -> int:
        return len(self.elements)

    @property
    def r_cutoff(self) -> float:
        """
        Return the maximum cutoff radius found between all descriptors.
        """
        return max([pot.descriptor.r_cutoff for pot in self.atomic_potential.values()])

    @property
    def descriptor(self) -> Dict:
        return {elem: pot.descriptor for elem, pot in self.atomic_potential.items()}

    @property
    def scaler(self) -> Dict:
        return {elem: pot.scaler for elem, pot in self.atomic_potential.items()}

    @property
    def model(self) -> Dict:
        return {elem: pot.model for elem, pot in self.atomic_potential.items()}

    # @property
    # def model_params(self) -> Dict:
    #     return {elem: pot.model_params for elem, pot in self.atomic_potential.items()}
