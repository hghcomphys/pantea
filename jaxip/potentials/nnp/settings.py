from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Union

from pydantic import Field, ValidationError, validator

from jaxip.atoms.element import ElementMap
from jaxip.config import _CFG
from jaxip.logger import logger
from jaxip.types import Element
from jaxip.utils.tokenize import tokenize


class RadialSymFuncArgs(NamedTuple):
    """Symmetry function arguments."""

    central_element: str
    acsf_type: int
    neighbor_element_j: str
    eta: float
    r_cutoff: float
    r_shift: float


class AngularSymFuncArgs(NamedTuple):
    """Symmetry function arguments."""

    central_element: str
    acsf_type: int
    neighbor_element_j: str
    eta: float
    r_cutoff: float
    r_shift: float
    neighbor_element_k: str
    lambda0: float
    zeta: float


SymFuncArgs = Union[RadialSymFuncArgs, AngularSymFuncArgs]


cutoff_function_map: Mapping[str, str] = {
    "0": "hard",
    "1": "cos",
    "2": "tanhu",
    "3": "tanh",
    "4": "exp",
    "5": "poly1",
    "6": "poly2",
}

scaler_type_map: Mapping[str, str] = {
    "center_symmetry_functions": "center",
    "scale_symmetry_functions": "scale",
    "scale_center_symmetry_functions": "scale_center",
    "scale_center_symmetry_functions_sigma": "scale_center_sigma",
}

activation_function_map: Mapping[str, str] = {
    "l": "identity",
    "t": "tanh",
    "s": "logistic",
    "p": "softplus",
    "r": "relu",
    "g": "gaussian",
    "c": "cos",
    "e": "exp",
    "h": "harmonic",
}

updater_type_map: Mapping[str, str] = {
    "0": "gradient_descent",
    "1": "kalman_filter",
}

gradient_type_map: Mapping[str, str] = {
    "0": "fixed_step",
    "1": "adam",
}


class NeuralNetworkPotentialSettings(_CFG):
    """
    A configuration class for neural network potential parameters.

    It contains a collections of all potential setting keywords and their default values.
    """

    # sphinx-pydantic can be useful to better document pydantic classes
    # https://pypi.org/project/autodoc-pydantic/0.1.1/

    # General
    random_seed: int = Field(
        2023, description="determine initialize neural network weights."
    )

    number_of_elements: int
    elements: List[Element] = Field(default_factory=list)
    atom_energy: Dict[Element, float] = Field(default_factory=dict)
    scaler_save_naming_format: str = "scaling.{:03d}.data"
    model_save_naming_format: str = "weights.{:03d}.pkl"
    # Neural Network
    weights_min: float = -1.0
    weights_max: float = 1.0
    global_hidden_layers_short: int
    global_nodes_short: List[int]
    global_activation_short: List[str]
    # Trainer
    epochs: int = 1
    updater_type: str = "gradient_descent"
    gradient_type: str = "adam"
    main_error_metric: str = "RMSE"
    force_weight: float = 1.0
    short_force_fraction: float = 0.1
    short_energy_fraction: float = 1.0
    test_fraction: float = 0.1
    save_best_model: bool = True
    gradient_eta: float = 1.0e-5
    gradient_adam_eta: float = 1.0e-3
    gradient_adam_beta1: float = 0.9
    gradient_adam_beta2: float = 0.999
    gradient_adam_epsilon: float = 1.0e-8
    gradient_adam_weight_decay: float = 1.0e-4
    kalman_type: int = 0
    kalman_epsilon: float = 0.01
    kalman_q0: float = 0.01
    kalman_qtau: float = 2.302
    kalman_qmin: float = 1.0e-6
    kalman_eta: float = 0.01
    kalman_etatau: float = 2.302
    kalman_etamax: float = 1.0
    kalman_lambda_short: float = 0.96000
    kalman_neu_short: float = 0.99950
    # Symmetry Function
    cutoff_type: str = "tanh"
    scale_type: str = "center"
    scale_min_short: float = 0.0
    scale_max_short: float = 1.0
    symfunction_short: List[SymFuncArgs] = Field(default_factory=list)

    @classmethod
    def from_file(cls, filename: Path) -> NeuralNetworkPotentialSettings:
        """
        Read all potential settings from the input file.

        Parameters such as `elements`, `cutoff type`, `symmetry functions`, `neural network`, `training parameters`, etc
        (see `here <https://compphysvienna.github.io/n2p2/topics/keywords.html>`_).
        """

        logger.info(f"Reading potential settings: {str(filename)}")
        dict_: Dict = cls._read_from_file(filename)
        kwargs: Dict[str, Any] = dict()
        kwargs["atom_energy"] = dict()
        kwargs["symfunction_short"] = list()

        for line_keyword, tokens in dict_.items():
            keyword: str = "_".join(line_keyword.split("_")[1:])  # type: ignore
            # ------------- General -------------
            if keyword == "number_of_elements":  # this keyword can be ignored
                kwargs[keyword] = tokens[0]
            elif keyword == "elements":
                kwargs[keyword] = sorted(
                    set([t for t in tokens]), key=ElementMap.get_atomic_number
                )
            elif keyword == "atom_energy":
                kwargs[keyword].update({tokens[0]: tokens[1]})
            elif keyword == "random_seed":
                kwargs[keyword] = tokens[0]
            # ------------- Neural Network -------------
            elif keyword == "global_hidden_layers_short":
                kwargs[keyword] = tokens[0]
            elif keyword == "global_nodes_short":
                kwargs[keyword] = [t for t in tokens]
            elif keyword == "global_activation_short":
                kwargs[keyword] = [activation_function_map[t] for t in tokens]
            elif keyword == "weights_min":
                kwargs[keyword] = tokens[0]
            elif keyword == "weights_max":
                kwargs[keyword] = tokens[0]
            # ------------- Trainer -------------
            elif keyword == "main_error_metric":
                kwargs[keyword] = tokens[0]
            elif keyword == "epochs":
                kwargs[keyword] = tokens[0]
            elif keyword == "test_fraction":
                kwargs[keyword] = tokens[0]
            elif keyword == "updater_type":
                kwargs[keyword] = updater_type_map[tokens[0]]
            elif keyword == "gradient_type":
                kwargs[keyword] = gradient_type_map[tokens[0]]
            elif keyword == "gradient_eta":
                kwargs[keyword] = tokens[0]
            elif keyword == "gradient_adam_eta":
                kwargs[keyword] = tokens[0]
            elif keyword == "gradient_adam_beta1":
                kwargs[keyword] = tokens[0]
            elif keyword == "gradient_adam_beta2":
                kwargs[keyword] = tokens[0]
            elif keyword == "gradient_adam_epsilon":
                kwargs[keyword] = tokens[0]
            elif keyword == "gradient_adam_weight_decay":
                kwargs[keyword] = tokens[0]
            elif keyword == "force_weight":
                kwargs[keyword] = tokens[0]
            elif keyword == "short_force_fraction":
                kwargs[keyword] = tokens[0]
            elif keyword == "short_energy_fraction":
                kwargs[keyword] = tokens[0]
            elif keyword == "kalman_type":
                kwargs[keyword] = tokens[0]
            elif keyword == "kalman_epsilon":
                kwargs[keyword] = tokens[0]
            elif keyword == "kalman_q0":
                kwargs[keyword] = tokens[0]
            elif keyword == "kalman_qtau":
                kwargs[keyword] = tokens[0]
            elif keyword == "kalman_qmin":
                kwargs[keyword] = tokens[0]
            elif keyword == "kalman_eta":
                kwargs[keyword] = tokens[0]
            elif keyword == "kalman_etatau":
                kwargs[keyword] = tokens[0]
            elif keyword == "kalman_etamax":
                kwargs[keyword] = tokens[0]
            elif keyword == "kalman_lambda_short":
                kwargs[keyword] = tokens[0]
            elif keyword == "kalman_neu_short":
                kwargs[keyword] = tokens[0]
            # ------------- Symmetry Function -------------
            elif keyword == "cutoff_type":
                kwargs[keyword] = cutoff_function_map[tokens[0]]
            elif keyword == "center_symmetry_functions":
                kwargs["scale_type"] = scaler_type_map[keyword]
            elif keyword == "scale_symmetry_functions":
                kwargs["scale_type"] = scaler_type_map[keyword]
            elif keyword == "scale_center_symmetry_functions":
                kwargs["scale_type"] = scaler_type_map[keyword]
            elif keyword == "scale_center_symmetry_functions_sigma":
                kwargs["scale_type"] = scaler_type_map[keyword]
            elif keyword == "scale_min_short":
                kwargs[keyword] = tokens[0]
            elif keyword == "scale_max_short":
                kwargs[keyword] = tokens[0]
            elif keyword == "symfunction_short":
                acsf_type = int(tokens[1])
                args: SymFuncArgs
                if acsf_type < 3:  # radial
                    args = RadialSymFuncArgs(
                        central_element=tokens[0],
                        acsf_type=acsf_type,
                        neighbor_element_j=tokens[2],
                        eta=float(tokens[3]),
                        r_shift=float(tokens[4]),
                        r_cutoff=float(tokens[5]),
                    )
                else:  # angular
                    args = AngularSymFuncArgs(
                        central_element=tokens[0],
                        acsf_type=acsf_type,
                        neighbor_element_j=tokens[2],
                        neighbor_element_k=tokens[3],
                        eta=float(tokens[4]),
                        lambda0=float(tokens[5]),
                        zeta=float(tokens[6]),
                        r_cutoff=float(tokens[7]),
                        r_shift=float(tokens[8]) if len(tokens) == 9 else 0.0,
                    )
                kwargs[keyword].append(args)
        try:
            settings = cls(**kwargs)
        except ValidationError as e:
            logger.error(str(e), exception=ValueError)
        return settings  # type: ignore

    @classmethod
    def _read_from_file(cls, filename) -> Dict:
        """Read all keywords from the input setting file."""
        dict_ = dict()
        n_line: int = 0
        with open(str(filename), "r") as file:
            while True:
                line = file.readline()
                n_line += 1
                if not line:
                    break
                # Read keyword and values
                keyword, tokens = tokenize(line, comment="#")  # type: ignore
                if keyword is not None:
                    if keyword not in cls.__annotations__:
                        logger.debug(f"Unknown keyword (line {n_line}):'{keyword}'")
                    else:
                        logger.debug(f"keyword:'{keyword}', tokens:{tokens}")
                        dict_[f"line{n_line:04d}_{keyword}"] = tokens
        return dict_

    @validator("elements")
    def number_of_elements_match(cls, v, values) -> Any:
        if "number_of_elements" in values and len(v) != values["number_of_elements"]:
            raise ValueError("number of elements is not consistent")
        return v

    @validator("test_fraction")
    def test_fraction_range(cls, v) -> Any:
        assert 0.0 <= v <= 1.0, "must be between [0, 1]"
        return v
