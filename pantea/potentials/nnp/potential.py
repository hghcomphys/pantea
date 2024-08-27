from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Protocol, Tuple

import jax.numpy as jnp
from frozendict import frozendict
from jax import random
from tqdm import tqdm

from pantea.atoms.element import ElementMap
from pantea.atoms.structure import Structure
from pantea.datasets.dataset import Dataset
from pantea.descriptors.acsf.acsf import ACSF
from pantea.descriptors.acsf.angular import G3, G9
from pantea.descriptors.acsf.cutoff import CutoffFunction
from pantea.descriptors.acsf.radial import G1, G2
from pantea.descriptors.acsf.symmetry import NeighborElements
from pantea.descriptors.scaler import DescriptorScaler
from pantea.logger import logger
from pantea.models.nn.initializer import UniformInitializer
from pantea.models.nn.network import NeuralNetworkModel
from pantea.potentials.nnp.atomic_potential import AtomicPotential, ModelParams
from pantea.potentials.nnp.energy import _compute_energy
from pantea.potentials.nnp.force import _compute_forces
from pantea.potentials.nnp.gradient_descent import GradientDescentUpdater
from pantea.potentials.nnp.kalman_filter import KalmanFilterUpdater
from pantea.potentials.nnp.settings import NeuralNetworkPotentialSettings
from pantea.types import Array, Element


class UpdaterInterface(Protocol):
    """Interface for potential weight updaters."""

    def fit(self, dataset: Dataset) -> Dict: ...


@dataclass
class NeuralNetworkPotential:
    """
    High-dimensional neural network potential (HDNNP) - second generation.

    It contains all the required descriptors, scalers, and neural networks for each element,
    and an updater to fit the potential model using the reference structure data.
    """

    directory: Path
    settings: NeuralNetworkPotentialSettings
    atomic_potentials: frozendict[Element, AtomicPotential]
    models_params: Dict[Element, ModelParams]
    # updater: Optional[UpdaterInterface] = field(default=None, repr=False, init=False)

    @classmethod
    def from_runner(
        cls,
        directory: Path,
        potential_filename: str = "input.nn",
    ) -> NeuralNetworkPotential:
        logger.info(f"Creating potential from RuNNer: {str(directory)}")
        potfile = Path(directory) / potential_filename
        logger.info(f"Initializing potential settings from RuNNer file: {potfile.name}")
        settings = NeuralNetworkPotentialSettings.from_file(potfile)
        atomic_potentials = cls._build_atomic_potentials(settings)
        models_params = cls._initialize_models_params(settings, atomic_potentials)
        return NeuralNetworkPotential(
            directory=directory,
            settings=settings,
            atomic_potentials=atomic_potentials,
            models_params=models_params,
        )

    @classmethod
    def from_json(
        cls,
        filename: Path,
        output_dir: Path = Path("./"),
    ) -> NeuralNetworkPotential:
        logger.info(f"Creating potential from JSON file: {str(filename)}")
        logger.info(f"Potential output directory: {str(output_dir)}")
        settings = NeuralNetworkPotentialSettings.from_json(filename)
        atomic_potentials = cls._build_atomic_potentials(settings)
        models_params = cls._initialize_models_params(settings, atomic_potentials)
        return NeuralNetworkPotential(
            directory=output_dir,
            settings=settings,
            atomic_potentials=atomic_potentials,
            models_params=models_params,
        )

    def __call__(self, structure: Structure) -> Array:
        """
        Compute the total energy.

        :param structure: Structure
        :return: total energy
        """
        return _compute_energy(
            self.atomic_potentials,
            structure.get_positions_per_element(),
            self.models_params,
            structure.as_kernel_args(),
        )  # type: ignore

    def compute_forces(self, structure: Structure) -> Array:
        """
        Compute force components.

        :param structure: input structure
        :return: predicted force components for all atoms
        """
        forces_dict = _compute_forces(
            self.atomic_potentials,
            structure.get_positions_per_element(),
            self.models_params,
            structure.as_kernel_args(),
        )
        forces = jnp.empty_like(structure.forces)
        for element in structure.get_unique_elements():
            atom_index = structure.select(element)
            forces = forces.at[atom_index].set(forces_dict[element])
        return forces

    def __post_init__(self) -> None:
        """
        Initialize potential components such as atomic potential, model params,
        and updater from the potential settings.
        """
        logger.info(f"Initializing {self.__class__.__name__}")
        logger.info(f"Number of elements: {self.num_elements}")
        for element in self.elements:
            logger.info(
                f"Element: {element} ({ElementMap.get_atomic_number_from_element(element)})"
            )



    @classmethod
    def _build_atomic_potentials(
        cls,
        settings: NeuralNetworkPotentialSettings,
    ) -> frozendict[Element, AtomicPotential]:
        """
        Initialize atomic potential for each element.
        This method can be override in case that different atomic potential is used.
        """
        logger.info("Initializing atomic potentials")
        atomic_potentials: Dict[Element, AtomicPotential] = dict()
        descriptors = cls._build_descriptors(settings)
        scalers = cls._build_scalers(settings)
        models = cls._build_models(settings)
        for element in settings.elements:
            atomic_potentials[element] = AtomicPotential(
                descriptor=descriptors[element],
                scaler=scalers[element],
                model=models[element],
            )
        return frozendict(atomic_potentials)

    @classmethod
    def _initialize_models_params(
        cls,
        settings: NeuralNetworkPotentialSettings,
        atomic_potentials: frozendict[Element, AtomicPotential],
    ) -> Dict[Element, ModelParams]:
        """
        Initialize neural network model parameters for each element
        (e.g. neural network kernel and bias values).

        This method be used to initialize model params of the potential with a different random seed.
        """
        logger.info("Initializing model params")
        models_params: Dict[Element, ModelParams] = dict()
        random_keys = random.split(
            random.PRNGKey(settings.random_seed),
            settings.number_of_elements,
        )
        for i, element in enumerate(settings.elements):
            models_params[element] = atomic_potentials[element].model.init(  # type: ignore
                random_keys[i],
                jnp.ones((1, atomic_potentials[element].model_input_size)),
            )[
                "params"
            ]
        return models_params

    @classmethod
    def _build_descriptors(
        cls,
        settings: NeuralNetworkPotentialSettings,
    ) -> Dict[Element, ACSF]:
        """Initialize descriptor for each element."""
        logger.info("Initializing descriptors")
        descriptors: Dict[Element, ACSF] = dict()
        radials = defaultdict(list)
        angulars = defaultdict(list)

        for args in settings.symfunction_short:

            cfn = CutoffFunction.from_type(
                cutoff_type=settings.cutoff_type,
                r_cutoff=args.r_cutoff,
            )

            if args.acsf_type == 1:
                symmetry_function = G1(cfn)
                neighbor_elements = NeighborElements(args.neighbor_element_j)
                radials[args.central_element].append(
                    (symmetry_function, neighbor_elements)
                )
            elif args.acsf_type == 2:
                symmetry_function = G2(cfn, eta=args.eta, r_shift=args.r_shift)
                neighbor_elements = NeighborElements(args.neighbor_element_j)
                radials[args.central_element].append(
                    (symmetry_function, neighbor_elements)
                )
            elif args.acsf_type == 3:
                symmetry_function = G3(
                    cfn,
                    eta=args.eta,
                    zeta=args.zeta,  # type: ignore
                    lambda0=args.lambda0,  # type: ignore
                    r_shift=args.r_cutoff,
                )
                neighbor_elements = NeighborElements(
                    args.neighbor_element_j, args.neighbor_element_k
                )
                angulars[args.central_element].append(
                    (symmetry_function, neighbor_elements)
                )
            elif args.acsf_type == 9:
                symmetry_function = G9(
                    cfn,
                    eta=args.eta,
                    zeta=args.zeta,  # type: ignore
                    lambda0=args.lambda0,  # type: ignore
                    r_shift=args.r_cutoff,
                )
                neighbor_elements = NeighborElements(
                    args.neighbor_element_j, args.neighbor_element_k
                )
                angular[args.central_element].append(
                    (symmetry_function, neighbor_elements)
                )

        # Instantiate ACSF for each element
        for element in settings.elements:
            descriptors[element] = ACSF(
                central_element=element,
                radial_symmetry_functions=tuple(radials[element]),
                angular_symmetry_functions=tuple(angulars[element]),
            )

        return descriptors

    @classmethod
    def _build_scalers(
        cls,
        settings: NeuralNetworkPotentialSettings,
    ) -> Dict[Element, DescriptorScaler]:
        """Initialize descriptor scaler for each element."""
        logger.info("Initializing descriptor scalers")
        scalers: Dict[Element, DescriptorScaler] = dict()
        # Prepare scaler input argument if exist in settings
        scaler_kwargs = {
            first: settings[second]
            for first, second in {
                "scale_type": "scale_type",
                "scale_min": "scale_min_short",
                "scale_max": "scale_max_short",
            }.items()
            if second in settings.keywords()
        }
        logger.debug(f"Scaler kwargs={scaler_kwargs}")
        # Assign an ACSF scaler to each element
        for element in settings.elements:
            scalers[element] = DescriptorScaler(**scaler_kwargs)
        return scalers

    @classmethod
    def _build_models(
        cls,
        settings: NeuralNetworkPotentialSettings,
    ) -> Dict[Element, NeuralNetworkModel]:
        """Initialize neural network model for each element."""
        logger.info("Initializing neural network models")
        models: Dict[Element, NeuralNetworkModel] = dict()

        for element in settings.elements:
            logger.debug(f"Element: {element}")

            hidden_layers = zip(
                settings.global_nodes_short,
                settings.global_activation_short[:-1],
            )
            output_layer: Tuple[int, str] = (
                1,
                settings.global_activation_short[-1],
            )
            kernel_initializer: UniformInitializer = UniformInitializer(
                weights_range=(
                    settings.weights_min,
                    settings.weights_max,
                )
            )
            models[element] = NeuralNetworkModel(
                hidden_layers=tuple([(n, t) for n, t in hidden_layers]),
                output_layer=output_layer,
                kernel_initializer=kernel_initializer,
            )
        return models

    # def _init_updater(self) -> None:
    #     """Initialize selected updater from the potential settings."""
    #     logger.info("Initializing updater")
    #     updater_type: str = self.settings.updater_type
    #     if updater_type == "kalman_filter":
    #         self.updater = KalmanFilterUpdater(potential=self)
    #     elif updater_type == "gradient_descent":
    #         self.updater = GradientDescentUpdater(potential=self)
    #     else:
    #         logger.error(
    #             f"Unknown updater type: {updater_type}",
    #             exception=TypeError,
    #         )
    #     logger.info(f"Updater type: {updater_type}")

    # ------------------------------------------------------------------------

    def fit_scaler(self, dataset: Dataset) -> None:
        """
        Fit scaler parameters for each element using the input structure data.
        No gradient history is required here.
        """
        print("Fitting descriptor scaler...")
        try:
            dataset_size: int = len(dataset)
            for index in tqdm(range(dataset_size)):
                structure: Structure = dataset[index]
                elements: Tuple[Element, ...] = structure.get_unique_elements()
                for element in elements:
                    x: Array = self.atomic_potentials[element].descriptor(structure)
                    self.atomic_potentials[element].scaler.fit(x)
        except KeyboardInterrupt:
            print("Keyboard Interrupt")
        else:
            print("Done.")

    # def fit_model(self, dataset: Dataset) -> Dict:
    #     """
    #     Fit energy model for all elements using the input structure loader.
    #     """
    #     # kwargs["validation_split"] = kwargs.get(
    #     #     "validation_split", self.settings.test_fraction
    #     # )
    #     # kwargs["epochs"] = kwargs.get("epochs", self.settings.epochs)
    #     history = defaultdict(list)
    #     print("Training potential...")
    #     try:
    #         history = self.updater.fit(dataset)  # type: ignore
    #     except KeyboardInterrupt:
    #         print("Keyboard Interrupt")
    #     else:
    #         print("Done.")
    #     return history

    def fit(self) -> None:
        """
        This method provides a user-friendly interface to fit both descriptor and model in one step.
        """
        ...

    # ------------------------------------------------------------------------

    def save_scaler(self) -> None:
        """
        This method saves scaler parameters for each element into separate files.
        """
        # Save scaler parameters for each element separately
        for element in self.settings.elements:
            atomic_number: int = ElementMap.get_atomic_number_from_element(element)
            scaler_file = Path(
                self.directory,
                self.settings.scaler_save_naming_format.format(atomic_number),
            )
            logger.info(
                f"Saving scaler parameters for element ({element}): {scaler_file.name}"
            )
            self.atomic_potentials[element].scaler.save(scaler_file)

    def save_model(self) -> None:
        """
        Save model weights separately for all elements.
        """
        for element in self.elements:
            atomic_number = ElementMap.get_atomic_number_from_element(element)
            model_file = Path(
                self.directory,
                self.settings.model_save_naming_format.format(atomic_number),
            )
            logger.info(
                f"Saving model weights for element ({element}): {model_file.name}"
            )
            self.atomic_potentials[element].model.save(
                model_file, self.models_params[element]
            )

    def save(self) -> None:
        """Save scaler and model into corresponding files for each element."""
        self.save_scaler()
        self.save_model()

    def load_scaler(self) -> None:
        """
        This method loads scaler parameters of each element from separate files.
        This save computational time as the would be no need to fit the scalers each time.
        """
        # Load scaler parameters for each element separately
        for element in self.elements:
            atomic_number = ElementMap.get_atomic_number_from_element(element)
            scaler_file = Path(
                self.directory,
                self.settings.scaler_save_naming_format.format(atomic_number),
            )
            logger.info(
                f"Loading scaler parameters for element ({element}): {scaler_file.name}"
            )
            self.atomic_potentials[element].scaler.load(scaler_file)

    def load_model(self) -> None:
        """
        Load model weights separately for all elements.
        """
        for element in self.elements:
            atomic_number = ElementMap.get_atomic_number_from_element(element)
            model_file = Path(
                self.directory,
                self.settings.model_save_naming_format.format(atomic_number),
            )
            logger.info(
                f"Loading model weights for element ({element}): {model_file.name}"
            )
            self.models_params[element] = self.atomic_potentials[element].model.load(
                model_file
            )

    def load(self) -> None:
        """Load element scaler and model from their corresponding files."""
        self.load_scaler()
        self.load_model()

    def set_extrapolation_warnings(self, threshold: Optional[int] = None) -> None:
        """
        shows warning whenever a descriptor value is out of bounds defined by
        minimum/maximum values in the scaler.

        set_extrapolation_warnings(None) will disable it.

        :param threshold: maximum number of warnings
        :type threshold: int
        """
        logger.info(f"Setting extrapolation warning: {threshold}")
        for pot in self.atomic_potentials.values():
            pot.scaler.set_max_number_of_warnings(threshold)

    @property
    def extrapolation_warnings(self) -> Dict[Element, int]:
        return {
            element: pot.scaler.number_of_warnings
            for element, pot in self.atomic_potentials.items()
        }

    @property
    def r_cutoff(self) -> float:
        """Return the maximum cutoff radius found between all descriptors."""
        return max([pot.descriptor.r_cutoff for pot in self.atomic_potentials.values()])

    @property
    def descriptors(self) -> Dict[Element, ACSF]:
        """Return descriptor for each element."""
        return {elem: pot.descriptor for elem, pot in self.atomic_potentials.items()}

    @property
    def scalers(self) -> Dict[Element, DescriptorScaler]:
        """Return scaler for each element."""
        return {elem: pot.scaler for elem, pot in self.atomic_potentials.items()}

    @property
    def models(self) -> Dict[Element, NeuralNetworkModel]:
        """Return model for each element."""
        return {elem: pot.model for elem, pot in self.atomic_potentials.items()}

    @property
    def elements(self) -> Tuple[Element, ...]:
        return tuple(element for element in self.settings.elements)

    @property
    def num_elements(self) -> int:
        return self.settings.number_of_elements


NNP = NeuralNetworkPotential
