from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Literal, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from pantea.logger import logger
from pantea.types import Array, default_dtype


class ScalerParams(NamedTuple):
    """Scaler statistical parameters."""

    dimension: Array
    nsamples: Array
    mean: Array
    sigma: Array
    minval: Array
    maxval: Array


class ScalerWarnings(NamedTuple):
    """Outlier check based on number of warnings."""

    number_of_warnings: int
    max_number_of_warnings: int


class ScaleRange(NamedTuple):
    """Expected range of scaled values."""

    min_value: Array
    max_value: Array


ScaleTransform = Callable[[ScalerParams, Array, ScaleRange], Array]
ScaleType = Literal["center", "scale", "scale_center", "scale_center_sigma"]


class DescriptorScaler:
    """
    Scale descriptor values between a given range.

    Scaling parameters are calculated by fitting over the samples in the dataset.
    Available scaler information are as follows:

    This descriptor scaler is also used to warn when setting out-of-distribution samples base
    on the fitted scaler parameters.
    """

    def __init__(self, scale_range: ScaleRange, transform: ScaleTransform) -> None:
        # using @dataclass(frozen=True) doesn't create hash!
        self.scale_range = scale_range
        self.transform = transform

    @classmethod
    def from_type(
        cls,
        scale_type: ScaleType,
        scale_min: float = 0.0,
        scale_max: float = 1.0,
    ) -> DescriptorScaler:
        """Initialize scaler including scaler type and min/max values."""
        if not (scale_min < scale_max):
            logger.error("Unexpected scale range values", exception=ValueError)
        scale_range = ScaleRange(
            min_value=jnp.array(scale_min),
            max_value=jnp.array(scale_max),
        )
        _SCALER_MAP_FUNC: Dict[ScaleType, Transform] = {
            "center": _center,
            "scale": _scale,
            "scale_center": _scale_center,
            "scale_center_sigma": _scale_center_sigma,
        }
        return cls(
            scale_range=scale_range,
            transform=_SCALER_MAP_FUNC[scale_type],
        )

    @classmethod
    def fit(cls, data: Array) -> ScalerParams:
        return _fit(data)

    @classmethod
    def partial_fit(cls, params: ScalerParams, data: Array) -> ScalerParams:
        """
        Partial fit scaler parameters.

        This is intended for cases when fit is not feasible due to very large number of samples
        (see `this`_ for more details).

        .. _this: https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        """
        return _partial_fit(params, jnp.atleast_2d(data))

    def __call__(self, params: ScalerParams, data: Array) -> Array:
        """Transform the input descriptor values using the scaler parameters."""
        return self.transform(params, jnp.atleast_2d(data), self.scale_range)

    @classmethod
    def initialize_warnings(
        cls,
        number_of_warnings: int = 0,
        max_number_of_warnings: int = -1,
    ) -> ScalerWarnings:
        return ScalerWarnings(number_of_warnings, max_number_of_warnings)

    @classmethod
    def check_warnings(
        cls,
        params: ScalerParams,
        data: Array,
        warnings: ScalerWarnings,
    ) -> ScalerWarnings:
        """
        Check whether the output scaler values exceed the predefined min/max range values or not.

        If it's the case, it keeps counting the number of warnings and
        raises an error when it exceeds the maximum number.

        An out of range descriptor value is an indication of
        the descriptor extrapolation which has to be avoided.
        """
        if warnings.max_number_of_warnings < 0:
            return warnings

        new_warnings = ScalerWarnings(
            warnings.number_of_warnings
            + int(_calculate_number_of_warnings(params, data)),
            warnings.max_number_of_warnings,
        )
        if new_warnings.number_of_warnings >= new_warnings.max_number_of_warnings:
            logger.warning(
                "Exceeding maximum number scaler extrapolation warnings: "
                f"{new_warnings.number_of_warnings} "
                f"(max={new_warnings.max_number_of_warnings})"
            )
        return new_warnings

    @classmethod
    def save(
        cls,
        params: ScalerParams,
        filename: Path,
    ) -> None:
        """Save scaler parameters into file."""
        logger.debug(f"Saving scaler parameters into '{str(filename)}'")
        with open(str(filename), "w") as file:
            serialized_params = {k: v.tolist() for k, v in params._asdict().items()}
            json.dump(serialized_params, file, indent=4)

    @classmethod
    def load(
        cls,
        filename: Path,
        integer_type_keys: Tuple[str, ...] = ("dimension", "nsamples"),
    ) -> ScalerParams:
        """Load scaler parameters from file."""
        logger.debug(f"Loading scaler parameters from '{str(filename)}'")
        with open(str(filename), "r") as file:
            serialized_params = json.load(file)
            jnp_params = {
                k: jnp.asarray(v, dtype=default_dtype.FLOATX)
                for k, v in serialized_params.items()
                if k not in integer_type_keys
            }
            jnp_params.update(
                {
                    k: jnp.asarray(v, dtype=default_dtype.INT)
                    for k, v in serialized_params.items()
                    if k in integer_type_keys
                }
            )
            return ScalerParams(**jnp_params)

    @classmethod
    def _check_dimension(cls, params: ScalerParams, data: Array) -> Array:
        data = jnp.atleast_2d(data)  # type: ignore
        dimension = int(params.dimension)
        if data.shape[1] != dimension:
            logger.error(
                f"Data dimension doesn't match: {data.shape[1]} (expected {dimension})",
                exception=ValueError,
            )
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(transform='{self.transform.__name__}', "
            f"scale_range=({self.scale_min}, {self.scale_max}))"
        )

    @property
    def scale_min(self) -> float:
        return float(self.scale_range.min_value)

    @property
    def scale_max(self) -> float:
        return float(self.scale_range.max_value)


@jax.jit
def _center(
    params: ScalerParams,
    data: Array,
    scale_range: ScaleRange,
) -> Array:
    return data - params.mean


@jax.jit
def _scale(
    params: ScalerParams,
    data: Array,
    scale_range: ScaleRange,
) -> Array:
    return scale_range.min_value + (scale_range.max_value - scale_range.min_value) * (
        data - params.minval
    ) / (params.maxval - params.minval)


@jax.jit
def _scale_center(
    params: ScalerParams,
    data: Array,
    scale_range: ScaleRange,
) -> Array:
    return scale_range.min_value + (scale_range.max_value - scale_range.min_value) * (
        data - params.mean
    ) / (params.maxval - params.minval)


@jax.jit
def _scale_center_sigma(
    params: ScalerParams, data: Array, scale_range: ScaleRange
) -> Array:
    return (
        scale_range.min_value
        + (scale_range.min_value - scale_range.max_value)
        * (data - params.mean)
        / params.sigma
    )


@jax.jit
def _fit(data: Array) -> ScalerParams:
    return ScalerParams(
        dimension=_to_jax_int(data.shape[1]),
        nsamples=_to_jax_int(data.shape[0]),
        mean=jnp.mean(data, axis=0),
        sigma=jnp.std(data, axis=0),
        maxval=jnp.max(data, axis=0),
        minval=jnp.min(data, axis=0),
    )


@jax.jit
def _partial_fit(params: ScalerParams, data: Array) -> ScalerParams:
    # Calculate params for a new batch of data
    new_mean: Array = jnp.mean(data, axis=0)
    new_sigma: Array = jnp.std(data, axis=0)
    new_min: Array = jnp.min(data, axis=0)
    new_max: Array = jnp.max(data, axis=0)
    m, n = params.nsamples, data.shape[0]
    # Calculate scaler new params for the entire data
    fm, fn = m / (m + n), n / (m + n)
    mean_diff = params.mean - new_mean
    mean = fm * params.mean + fn * new_mean
    sigma = jnp.sqrt(
        (fm * params.sigma) * params.sigma
        + (fn * new_sigma) * new_sigma
        + (fm * mean_diff) * (fn * mean_diff)
    )  # split coefficients due to integer/float overflow
    # Find min & max
    maxval = jnp.maximum(params.maxval, new_max)
    minval = jnp.minimum(params.minval, new_min)
    nsamples = params.nsamples + n
    # Return updated params
    return ScalerParams(params.dimension, nsamples, mean, sigma, minval, maxval)


@jax.jit
def _calculate_number_of_warnings(params: ScalerParams, data: Array) -> Array:
    if data.ndim == 2:
        gt = jax.lax.gt(data, params.maxval[None, :])
        lt = jax.lax.gt(params.minval[None, :], data)
    else:
        gt = jax.lax.gt(data, params.maxval)
        lt = jax.lax.gt(params.minval, data)
    # alternative counting is using sum
    return jnp.any(jnp.logical_or(gt, lt))


def _to_jax_int(value: int) -> Array:
    return jnp.array(value, dtype=default_dtype.INT)
