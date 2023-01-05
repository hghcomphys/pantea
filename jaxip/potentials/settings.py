from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from jaxip.logger import logger
from jaxip.structure.element import ElementMap
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
    """A container for neural network potential parameters."""

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

    def __getitem__(self, keyword: str) -> Any:
        """Get value for the input keyword argument."""
        return self._settings[keyword]

    def keywords(self):
        """Return list of existing keywords in the potential settings."""
        return self._settings.keys()
