from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import jax.numpy as jnp
from frozendict import frozendict
from jax import random
from jaxip.datasets.runner import RunnerStructureDataset
from jaxip.descriptors.acsf.acsf import ACSF
from jaxip.descriptors.acsf.angular import G3, G9
from jaxip.descriptors.acsf.cutoff import CutoffFunction
from jaxip.descriptors.acsf.radial import G1, G2
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
from tqdm import tqdm


class StaticArgs(NamedTuple):
    descriptor: Descriptor
    scaler: DescriptorScaler
    model: NeuralNetworkModel


@dataclass()
class PerElementPotential:
    descriptor: Descriptor
    scaler: DescriptorScaler
    model: NeuralNetworkModel
    model_params: frozendict = field(repr=False)


@dataclass
class NeuralNetworkPotential:
    """
    A suitcase class of high-dimensional neural network potential (HDNNP).

    It contains all the required descriptors, scalers, and neural networks for each element,
    and a trainer to fit the potential using reference structure data.
    """

    potfile: Path
    potential: Dict[str, PerElementPotential] = field(default_factory=dict)
    trainer: Optional[NeuralNetworkPotentialTrainer] = None

    def __post_init__(self) -> None:
        """Post initialize potential parameters."""
        # TODO: input settings externally (it conflicts with the potfile input)
        self.settings: NeuralNetworkPotentialSettings = NeuralNetworkPotentialSettings(
            filename=self.potfile
        )
        if len(self.potential) == 0:
            self._init_potential()

        if self.trainer is None:
            logger.debug("[Setting trainer]")
            self.trainer = NeuralNetworkPotentialTrainer(potential=self)

    def _init_potential(self) -> None:
        """Initialize potential for each element"""
        descriptor: Dict[str, Descriptor] = self._init_descriptor()
        scaler: Dict[str, DescriptorScaler] = self._init_scaler()
        model: Dict[str, NeuralNetworkModel] = self._init_model()
        model_params: Dict[str, frozendict] = self._init_model_params(model, descriptor)
        for element in self.elements:
            self.potential[element] = PerElementPotential(
                descriptor=descriptor[element],
                scaler=scaler[element],
                model=model[element],
                model_params=model_params[element],
            )

    def _init_descriptor(self) -> Dict[str, Descriptor]:
        """Initialize descriptor for each element."""
        descriptor: Dict[str, Descriptor] = dict()
        # Elements
        logger.info(f"Number of elements: {self.num_elements}")
        for element in self.elements:
            logger.info(f"Element: {element} ({ElementMap.get_atomic_number(element)})")
        # Instantiate ACSF for each element
        for element in self.elements:
            descriptor[element] = ACSF(element)
        # Add symmetry functions
        logger.debug(
            "Registering symmetry functions (radial and angular)"
        )  # TODO: move logging inside .add() method

        for cfg in self.settings["symfunction_short"]:
            if cfg[1] == 1:
                descriptor[cfg[0]].add(
                    symmetry_function=G1(
                        CutoffFunction(
                            r_cutoff=cfg[5], cutoff_type=self.settings["cutoff_type"]
                        )
                    ),
                    neighbor_element_j=cfg[2],
                )
            elif cfg[1] == 2:
                descriptor[cfg[0]].add(
                    symmetry_function=G2(
                        CutoffFunction(
                            r_cutoff=cfg[5], cutoff_type=self.settings["cutoff_type"]
                        ),
                        eta=cfg[3],
                        r_shift=cfg[4],
                    ),
                    neighbor_element_j=cfg[2],
                )
            elif cfg[1] == 3:
                descriptor[cfg[0]].add(
                    symmetry_function=G3(
                        CutoffFunction(
                            r_cutoff=cfg[7], cutoff_type=self.settings["cutoff_type"]
                        ),
                        eta=cfg[4],
                        zeta=cfg[6],
                        lambda0=cfg[5],
                        r_shift=0.0,
                    ),  # TODO: add r_shift!
                    neighbor_element_j=cfg[2],
                    neighbor_element_k=cfg[3],
                )
            elif cfg[1] == 9:
                descriptor[cfg[0]].add(
                    symmetry_function=G9(
                        CutoffFunction(
                            r_cutoff=cfg[7], cutoff_type=self.settings["cutoff_type"]
                        ),
                        eta=cfg[4],
                        zeta=cfg[6],
                        lambda0=cfg[5],
                        r_shift=0.0,
                    ),  # TODO: add r_shift!
                    neighbor_element_j=cfg[2],
                    neighbor_element_k=cfg[3],
                )
        return descriptor

    def _init_scaler(self) -> Dict[str, DescriptorScaler]:
        """Initialize descriptor scaler for each element."""
        scaler: Dict[str, DescriptorScaler] = dict()
        # Prepare scaler input argument if exist in settings
        scaler_kwargs = {
            first: self.settings[second]
            for first, second in {
                "scale_type": "scale_type",
                "scale_min": "scale_min_short",
                "scale_max": "scale_max_short",
            }.items()
            if second in self.settings.keywords()
        }
        logger.debug(f"Scaler kwargs={scaler_kwargs}")
        # Assign an ACSF scaler to each element
        for element in self.elements:
            scaler[element] = DescriptorScaler(**scaler_kwargs)
        return scaler

    def _init_model(self) -> Dict[str, NeuralNetworkModel]:
        """Initialize neural network model for each element."""
        model: Dict[str, NeuralNetworkModel] = dict()
        for element in self.elements:
            logger.debug(f"Element: {element}")
            # TODO: what if we have a different model architecture for each element
            hidden_layers = zip(
                self.settings["global_nodes_short"],
                self.settings["global_activation_short"][:-1],
            )
            output_layer: Tuple[int, str] = (
                1,
                self.settings["global_activation_short"][-1],
            )
            kernel_initializer: UniformInitializer = UniformInitializer(
                weights_range=(
                    self.settings["weights_min"],
                    self.settings["weights_max"],
                )
            )
            model[element] = NeuralNetworkModel(
                hidden_layers=tuple([(n, t) for n, t in hidden_layers]),
                output_layer=output_layer,
                kernel_initializer=kernel_initializer,
            )
        return model

    def _init_model_params(
        self,
        model: Dict[str, NeuralNetworkModel],
        descriptor: Dict[str, Descriptor],
    ) -> Dict[str, frozendict]:
        """Initialize neural network model parameters for each element using model and descriptor parameters."""
        model_params: Dict[str, frozendict] = dict()
        random_keys = random.split(random.PRNGKey(0), self.num_elements)
        for i, element in enumerate(self.elements):
            model_params[element] = model[element].init(  # type: ignore
                random_keys[i],
                jnp.ones((1, descriptor[element].num_descriptors)),
            )["params"]
        return model_params

    # @Profiler.profile
    def fit_scaler(self, dataset: RunnerStructureDataset, **kwargs) -> None:
        """
        Fit scaler parameters for each element using the input structure data.
        No gradient history is required here.
        """
        save_scaler: bool = kwargs.get("save_scaler", True)

        # loader = TorchDataLoader(dataset, collate_fn=lambda batch: batch)
        print("Fitting descriptor scalers...")
        for structure in tqdm(dataset):
            for element in structure.elements:
                aid: Array = structure.select(element)
                dsc_val = self.potential[element].descriptor(
                    structure, aid
                )  # FIXME: remove aid
                self.potential[element].scaler.fit(dsc_val)
        print("Done.\n")

        if save_scaler:
            self.save_scaler()

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
            self.potential[element].scaler.save(scaler_file)

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
            self.potential[element].scaler.load(scaler_file)

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
            self.potential[element].model.save(model_file)

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
            self.potential[element].model.load(model_file)

    def fit(self) -> None:
        """
        This method provides a user-friendly interface to fit both descriptor and model in one step.
        """
        pass

    def __call__(self, structure: Structure) -> Array:
        """
        Return the total energy of the input structure.

        :param structure: Structure
        :type structure: Structure
        :return: total energy
        :rtype: Array
        """
        return _energy_fn(
            self.get_static_args(),
            structure.get_positions(),
            self.model_params,
            structure.get_inputs(),
        )

    def compute_force(self, structure: Structure) -> Dict[str, Array]:
        forces: Dict[str, Array] = _compute_force(
            self.get_static_args(),
            structure.get_positions(),
            self.model_params,
            structure.get_inputs(),
        )
        return forces

    def get_static_args(self) -> frozendict:
        return frozendict(
            {
                element: StaticArgs(
                    self.potential[element].descriptor,
                    self.potential[element].scaler,
                    self.potential[element].model,
                )
                for element in self.elements
            }
        )

    def set_extrapolation_warnings(self, threshold: Optional[int] = None) -> None:
        """
        shows warning whenever a descriptor value is out of bounds defined by
        minimum/maximum values in the scaler.

        set_extrapolation_warnings(None) will disable it.

        :param threshold: maximum number of warnings
        :type threshold: int
        """
        logger.info(f"Setting extrapolation warning: {threshold}")
        for pot in self.potential.values():
            pot.scaler.set_max_number_of_warnings(threshold)

    @property
    def extrapolation_warnings(self) -> Dict[str, int]:
        return {
            element: pot.scaler.number_of_warnings
            for element, pot in self.potential.items()
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
        return max([pot.descriptor.r_cutoff for pot in self.potential.values()])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(potfile='{self.potfile.name}')"

    @property
    def descriptor(self) -> Dict:
        return {elem: pot.descriptor for elem, pot in self.potential.items()}

    @property
    def scaler(self) -> Dict:
        return {elem: pot.scaler for elem, pot in self.potential.items()}

    @property
    def model(self) -> Dict:
        return {elem: pot.model for elem, pot in self.potential.items()}

    @property
    def model_params(self) -> Dict:
        return {elem: pot.model_params for elem, pot in self.potential.items()}
