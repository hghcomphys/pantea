from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from jaxip.descriptors.acsf.acsf import ACSF
from jaxip.descriptors.acsf.angular import G3, G9
from jaxip.descriptors.acsf.cutoff import CutoffFunction
from jaxip.descriptors.acsf.radial import G1, G2
from jaxip.descriptors.base import Descriptor
from jaxip.descriptors.scaler import DescriptorScaler
from jaxip.logger import logger
from jaxip.models.initializer import UniformInitializer
from jaxip.models.nn import NeuralNetworkModel
from jaxip.structure.element import ElementMap
from jaxip.utils.attribute import set_as_attribute
from jaxip.utils.tokenize import tokenize

nnp_default_settings: Mapping[str, Any] = {
    "symfunction_short": list(),
    "epochs": 1,
    "updater_type": 0,
    "gradient_type": 1,
    "gradient_weight_decay": 1.0e-5,
    "main_error_metric": "RMSE",
    "weights_min": 0.0,
    "weights_max": 1.0,
    "test_fraction": 0.1,
    "force_weight": 1.0,
    "atom_energy": dict(),
    "scaler_save_naming_format": "scaling.{:03d}.data",
    "model_save_naming_format": "weights.{:03d}.zip",
    # TODO: add all default values
}

cutoff_function_map: Mapping[str, str] = {
    "0": "hard",
    "1": "cos",
    "2": "tanhu",
    "3": "tanh",
    "4": "exp",
    "5": "poly1",
    "6": "poly2",
    # TODO: poly 3 & 4
}

scaler_type_map: Mapping[str, str] = {
    "center_symmetry_functions": "center",
    "scale_symmetry_functions": "scale",
    "scale_center_symmetry_functions": "scale_center",
    "scale_center_symmetry_functions_sigma": "scale_center_sigma",
}


class NeuralNetworkPotentialSettings:
    """A configuration class for neural network potential parameters."""

    # FIXME: unite setting and configuration class

    def __init__(
        self,
        filename: Optional[Path] = None,
        default: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize potential settings."""
        self._settings: Dict[str, Any] = dict()

        self._settings.update(nnp_default_settings)
        if default is not None:
            self._settings.update(default)

        if filename is not None:
            self.read(filename)

        set_as_attribute(self, self._settings)

    def read(self, filename: Path) -> None:
        """
        Read all the potential settings from given input file.

        Parameters such as `elements`, `cutoff type`, `symmetry functions`, `neural network`, `training parameters`, etc
        (see `here <https://compphysvienna.github.io/n2p2/topics/keywords.html>`_).
        """
        # TODO: DRY, read keyword with map lambda function
        logger.debug("[Reading potential settings]")
        logger.debug(f"Potential file:'{str(filename)}'")

        with open(str(filename), "r") as file:
            while True:
                line = file.readline()
                if not line:
                    break

                # Read keyword and values
                keyword, tokens = tokenize(line, comment="#")
                if keyword is not None:
                    logger.debug(f"keyword:'{keyword}', values:{tokens}")

                # General settings
                if keyword == "number_of_elements":  # this keyword can be ignored
                    self._settings[keyword] = int(tokens[0])
                elif keyword == "elements":
                    self._settings[keyword] = sorted(
                        set([t for t in tokens]), key=ElementMap.get_atomic_number
                    )
                    assert self._settings["number_of_elements"] == len(
                        self._settings["elements"]
                    )
                elif keyword == "atom_energy":
                    self._settings[keyword].update({tokens[0]: float(tokens[1])})
                elif keyword == "cutoff_type":
                    self._settings[keyword] = cutoff_function_map[tokens[0]]
                elif keyword == "symfunction_short":
                    try:
                        acsf_ = (tokens[0], int(tokens[1]), tokens[2]) + tuple(
                            [float(t) for t in tokens[3:]]
                        )
                    except ValueError:
                        acsf_ = (
                            tokens[0],
                            int(tokens[1]),
                            tokens[2],
                            tokens[3],
                        ) + tuple([float(t) for t in tokens[4:]])
                    self._settings[keyword].append(acsf_)

                # Neural network
                elif keyword == "global_hidden_layers_short":
                    self._settings["global_hidden_layers_short"] = tokens[0]
                elif keyword == "global_nodes_short":
                    self._settings["global_nodes_short"] = [int(t) for t in tokens]
                elif keyword == "global_activation_short":
                    self._settings["global_activation_short"] = [t for t in tokens]
                elif keyword == "weights_min":
                    self._settings["weights_min"] = float(tokens[0])
                elif keyword == "weights_max":
                    self._settings["weights_max"] = float(tokens[0])

                # Symmetry function settings
                elif keyword == "center_symmetry_functions":
                    self._settings["scale_type"] = scaler_type_map[keyword]
                elif keyword == "scale_symmetry_functions":
                    self._settings["scale_type"] = scaler_type_map[keyword]
                elif keyword == "scale_center_symmetry_functions":
                    self._settings["scale_type"] = scaler_type_map[keyword]
                elif keyword == "scale_center_symmetry_functions_sigma":
                    self._settings["scale_type"] = scaler_type_map[keyword]
                elif keyword == "scale_min_short":
                    self._settings[keyword] = float(tokens[0])
                elif keyword == "scale_max_short":
                    self._settings[keyword] = float(tokens[0])

                # Trainer settings
                elif keyword == "main_error_metric":
                    self._settings[keyword] = tokens[0]
                elif keyword == "epochs":
                    self._settings[keyword] = int(tokens[0])
                elif keyword == "test_fraction":
                    self._settings[keyword] = float(tokens[0])
                elif keyword == "updater_type":
                    self._settings[keyword] = int(tokens[0])
                elif keyword == "gradient_type":
                    self._settings[keyword] = int(tokens[0])
                elif keyword == "gradient_eta":
                    self._settings[keyword] = float(tokens[0])
                elif keyword == "gradient_adam_eta":
                    self._settings[keyword] = float(tokens[0])
                elif keyword == "gradient_adam_beta1":
                    self._settings[keyword] = float(tokens[0])
                elif keyword == "gradient_adam_beta2":
                    self._settings[keyword] = float(tokens[0])
                elif keyword == "gradient_adam_epsilon":
                    self._settings[keyword] = float(tokens[0])
                elif keyword == "gradient_weight_decay":
                    self._settings[keyword] = float(tokens[0])
                elif keyword == "force_weight   ":
                    self._settings[keyword] = float(tokens[0])

    def get_descriptor(self) -> Dict[str, Descriptor]:
        """Initialize descriptor for each element."""
        descriptor: Dict[str, Descriptor] = dict()
        settings = self._settings
        # Elements
        logger.info(f"Number of elements: {settings['number_of_elements']}")
        for element in settings["elements"]:
            logger.info(f"Element: {element} ({ElementMap.get_atomic_number(element)})")
        # Instantiate ACSF for each element
        for element in settings["elements"]:
            descriptor[element] = ACSF(element)
        # Add symmetry functions
        logger.debug(
            "Registering symmetry functions (radial and angular)"
        )  # TODO: move logging inside .add() method

        for cfg in settings["symfunction_short"]:
            if cfg[1] == 1:
                descriptor[cfg[0]].add(
                    symmetry_function=G1(
                        CutoffFunction(
                            r_cutoff=cfg[5], cutoff_type=settings["cutoff_type"]
                        )
                    ),
                    neighbor_element_j=cfg[2],
                )
            elif cfg[1] == 2:
                descriptor[cfg[0]].add(
                    symmetry_function=G2(
                        CutoffFunction(
                            r_cutoff=cfg[5], cutoff_type=settings["cutoff_type"]
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
                            r_cutoff=cfg[7], cutoff_type=settings["cutoff_type"]
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
                            r_cutoff=cfg[7], cutoff_type=settings["cutoff_type"]
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

    def get_scaler(self) -> Dict[str, DescriptorScaler]:
        """Initialize descriptor scaler for each element."""
        scaler: Dict[str, DescriptorScaler] = dict()
        settings = self._settings
        # Prepare scaler input argument if exist in settings
        scaler_kwargs = {
            first: settings[second]
            for first, second in {
                "scale_type": "scale_type",
                "scale_min": "scale_min_short",
                "scale_max": "scale_max_short",
            }.items()
            if second in self.keywords()
        }
        logger.debug(f"Scaler kwargs={scaler_kwargs}")
        # Assign an ACSF scaler to each element
        for element in settings["elements"]:
            scaler[element] = DescriptorScaler(**scaler_kwargs)
        return scaler

    def get_model(self) -> Dict[str, NeuralNetworkModel]:
        """Initialize neural network model for each element."""
        model: Dict[str, NeuralNetworkModel] = dict()
        settings = self._settings
        for element in settings["elements"]:
            logger.debug(f"Element: {element}")
            # TODO: what if we have a different model architecture for each element
            hidden_layers = zip(
                settings["global_nodes_short"],
                settings["global_activation_short"][:-1],
            )
            output_layer: Tuple[int, str] = (
                1,
                settings["global_activation_short"][-1],
            )
            kernel_initializer: UniformInitializer = UniformInitializer(
                weights_range=(
                    settings["weights_min"],
                    settings["weights_max"],
                )
            )
            model[element] = NeuralNetworkModel(
                hidden_layers=tuple([(n, t) for n, t in hidden_layers]),
                output_layer=output_layer,
                kernel_initializer=kernel_initializer,
            )
        return model

    def __getitem__(self, keyword: str) -> Any:
        """Get value for the input keyword argument."""
        return self._settings[keyword]

    def keywords(self):
        """Return list of existing keywords in the potential settings."""
        return self._settings.keys()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
