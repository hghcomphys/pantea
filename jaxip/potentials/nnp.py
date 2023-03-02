from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
from frozendict import frozendict
from jax import random
from tqdm import tqdm

from jaxip.datasets.runner import RunnerStructureDataset
from jaxip.descriptors.acsf.acsf import ACSF
from jaxip.descriptors.acsf.angular import G3, G9
from jaxip.descriptors.acsf.cutoff import CutoffFunction
from jaxip.descriptors.acsf.radial import G1, G2
from jaxip.descriptors.scaler import DescriptorScaler
from jaxip.logger import logger
from jaxip.models.initializer import UniformInitializer
from jaxip.models.nn import NeuralNetworkModel
from jaxip.potentials._energy import _compute_energy
from jaxip.potentials._force import _compute_force
from jaxip.potentials.atomic_potential import AtomicPotential
from jaxip.potentials.kalman_filter import KalmanFilterTrainer as Trainer
from jaxip.potentials.settings import NeuralNetworkPotentialSettings as Settings
from jaxip.structure.element import ElementMap
from jaxip.structure.structure import Structure
from jaxip.types import Array, Element


@dataclass
class NeuralNetworkPotential:
    """
    High-dimensional neural network potential (HDNNP) - second generation.

    It contains all the required descriptors, scalers, and neural networks for each element,
    and a trainer to fit the potential using reference structure data.

    Example
    -------
    Ways to initialize neural network potential.

    .. code-block:: python
        :linenos:

        from jaxip.potentials import NeuralNetworkPotential
        from jaxip.potentials import NeuralNetworkPotentialSettings as Settings

        # Method #1 (potential file)
        nnp1 = NeuralNetworkPotential.create_from("input.nn")

        # Method #2 (dictionary of parameters)
        settings = Settings(**params_dict)
        nnp2 = NeuralNetworkPotential(settings)

        # Method #3 (json file)
        settings = Settings.from_json('h2o.json')
        nnp3 = NeuralNetworkPotential(settings)
    """

    settings: Settings = field(repr=False)
    output_dir: Path = field(default=Path("."), repr=False)
    atomic_potential: Dict[Element, AtomicPotential] = field(
        default_factory=dict, repr=True, init=False
    )
    model_params: Dict[Element, frozendict] = field(
        default_factory=dict, repr=False, init=False
    )
    trainer: Optional[Trainer] = field(default=None, repr=False, init=False)

    @classmethod
    def create_from(cls, potential_file: Path):
        """Create an instance of the potential from the input file (RuNNer format)."""
        potential_file = Path(potential_file)
        settings = Settings.create_from(potential_file)
        return NeuralNetworkPotential(
            settings=settings,
            output_dir=potential_file.parent,
        )

    def __post_init__(self) -> None:
        """
        Initialize potential components such as atomic potential, model params,
        and trainer from the potential settings.
        """
        if not self.atomic_potential:
            self._init_atomic_potential()
        if not self.model_params:
            self._init_model_params()
        if self.trainer is None:
            logger.debug("[Setting trainer]")
            self.trainer = Trainer(potential=self)

    def _init_atomic_potential(self) -> None:
        """
        Initialize atomic potential for each element.

        This method can be override in case that different atomic potential is used.
        """
        descriptor: Dict[Element, ACSF] = self._init_descriptor()
        scaler: Dict[Element, DescriptorScaler] = self._init_scaler()
        model: Dict[Element, NeuralNetworkModel] = self._init_model()
        for element in self.settings.elements:
            self.atomic_potential[element] = AtomicPotential(
                descriptor=descriptor[element],
                scaler=scaler[element],
                model=model[element],
            )

    def _init_model_params(self) -> None:
        """
        Initialize neural network model parameters for each element
        (e.g. neural network kernel and bias values).

        This method be used to initialize model params of the potential with a different random seed.
        """
        random_keys = random.split(
            random.PRNGKey(self.settings.random_seed),
            self.settings.number_of_elements,
        )
        for i, element in enumerate(self.settings.elements):
            self.model_params[element] = self.atomic_potential[element].model.init(  # type: ignore
                random_keys[i],
                jnp.ones((1, self.atomic_potential[element].model_input_size)),
            )[
                "params"
            ]

    def _init_descriptor(self) -> Dict[Element, ACSF]:
        """Initialize descriptor for each element."""
        descriptor: Dict[Element, ACSF] = dict()
        settings = self.settings
        # Elements
        logger.info(f"Number of elements: {settings.number_of_elements}")
        for element in settings.elements:
            logger.info(f"Element: {element} ({ElementMap.get_atomic_number(element)})")
        # Instantiate ACSF for each element
        for element in settings.elements:
            descriptor[element] = ACSF(element)
        # Add symmetry functions
        logger.debug(
            "Registering symmetry functions (radial and angular)"
        )  # TODO: move logging inside .add() method

        for args in settings.symfunction_short:
            if args.acsf_type == 1:
                descriptor[args.central_element].add(
                    symmetry_function=G1(
                        CutoffFunction(
                            r_cutoff=args.r_cutoff,
                            cutoff_type=settings.cutoff_type,
                        )
                    ),
                    neighbor_element_j=args.neighbor_element_j,
                )
            elif args.acsf_type == 2:
                descriptor[args.central_element].add(
                    symmetry_function=G2(
                        CutoffFunction(
                            r_cutoff=args.r_cutoff, cutoff_type=settings.cutoff_type
                        ),
                        eta=args.eta,
                        r_shift=args.r_shift,
                    ),
                    neighbor_element_j=args.neighbor_element_j,
                )
            elif args.acsf_type == 3:
                descriptor[args.central_element].add(
                    symmetry_function=G3(
                        CutoffFunction(
                            r_cutoff=args.r_cutoff, cutoff_type=settings.cutoff_type
                        ),
                        eta=args.eta,
                        zeta=args.zeta,  # type: ignore
                        lambda0=args.lambda0,  # type: ignore
                        r_shift=args.r_cutoff,
                    ),
                    neighbor_element_j=args.neighbor_element_j,
                    neighbor_element_k=args.neighbor_element_k,  # type: ignore
                )
            elif args.acsf_type == 9:
                descriptor[args.central_element].add(
                    symmetry_function=G9(
                        CutoffFunction(
                            r_cutoff=args.r_cutoff,
                            cutoff_type=settings.cutoff_type,
                        ),
                        eta=args.eta,
                        zeta=args.zeta,  # type: ignore
                        lambda0=args.lambda0,  # type: ignore
                        r_shift=args.r_shift,
                    ),
                    neighbor_element_j=args.neighbor_element_j,
                    neighbor_element_k=args.neighbor_element_k,  # type: ignore
                )
        return descriptor

    def _init_scaler(self) -> Dict[Element, DescriptorScaler]:
        """Initialize descriptor scaler for each element."""
        scaler: Dict[Element, DescriptorScaler] = dict()
        settings = self.settings
        # Prepare scaler input argument if exist in settings
        scaler_kwargs = {
            first: settings[second]
            for first, second in {
                "scale_type": "scale_type",
                "scale_min": "scale_min_short",
                "scale_max": "scale_max_short",
            }.items()
            if second in self.settings.keywords()
        }
        logger.debug(f"Scaler kwargs={scaler_kwargs}")
        # Assign an ACSF scaler to each element
        for element in settings.elements:
            scaler[element] = DescriptorScaler(**scaler_kwargs)
        return scaler

    def _init_model(self) -> Dict[Element, NeuralNetworkModel]:
        """Initialize neural network model for each element."""
        model: Dict[Element, NeuralNetworkModel] = dict()
        settings = self.settings
        for element in settings.elements:
            logger.debug(f"Element: {element}")
            # TODO: what if we have a different model architecture for each element
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
            model[element] = NeuralNetworkModel(
                hidden_layers=tuple([(n, t) for n, t in hidden_layers]),
                output_layer=output_layer,
                kernel_initializer=kernel_initializer,
            )
        return model

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
        try:
            for structure in tqdm(dataset):
                for element in structure.elements:
                    x: Array = self.atomic_potential[element].descriptor(structure)
                    self.atomic_potential[element].scaler.fit(x)
        except KeyboardInterrupt:
            print("Keyboard Interrupt")
        else:
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
        # kwargs["validation_split"] = kwargs.get(
        #     "validation_split", self.settings.test_fraction
        # )
        # kwargs["epochs"] = kwargs.get("epochs", self.settings.epochs)
        # FIXME: add trainer factory
        assert (
            self.settings.updater_type == "kalman_filter"
        ), "Only Kalman trainer is implemented"
        return self.trainer.train(dataset)  # , **kwargs)  # type: ignore

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
        for element in self.settings.elements:
            atomic_number: int = ElementMap.get_atomic_number(element)
            scaler_file = Path(
                self.output_dir,
                self.settings.scaler_save_naming_format.format(atomic_number),
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
                self.output_dir,
                self.settings.model_save_naming_format.format(atomic_number),
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
                self.output_dir,
                self.settings.scaler_save_naming_format.format(atomic_number),
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
                self.output_dir,
                self.settings.model_save_naming_format.format(atomic_number),
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

    @property
    def extrapolation_warnings(self) -> Dict[Element, int]:
        return {
            element: pot.scaler.number_of_warnings
            for element, pot in self.atomic_potential.items()
        }

    @property
    def elements(self) -> List[Element]:
        """Return elements."""
        return self.settings.elements

    @property
    def num_elements(self) -> int:
        """Return number of elements."""
        return self.settings.number_of_elements

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
