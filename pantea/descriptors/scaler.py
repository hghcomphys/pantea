from pathlib import Path
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from pantea.logger import logger
from pantea.types import Array, Dtype, default_dtype


def _to_jax_int(n: int) -> Array:
    return jnp.array(n, dtype=default_dtype.INT)


class ScalerParams(NamedTuple):
    """Scaler statistical parameters."""

    nsamples: Array = _to_jax_int(0)
    mean: Array = jnp.array([])
    sigma: Array = jnp.array([])
    minval: Array = jnp.array([])
    maxval: Array = jnp.array([])


class ScaleRange(NamedTuple):
    """Expected range of scaled values."""

    min_value: Array
    max_value: Array


class DescriptorScaler:
    """
    Scale descriptor values between a desired range.

    Scaling parameters are calculated by fitting over the samples in the dataset.
    Available scaler information are as follows:

    * mean: average values
    * sigma: standard deviation
    * maxval: maximum values
    * minval: minimum values

    This descriptor scaler is also used to warn when setting out-of-distribution samples base
    on the fitted scaler parameters.
    """

    def __init__(
        self,
        scale_type: str = "scale_center",
        scale_min: float = 0.0,
        scale_max: float = 1.0,
    ) -> None:
        """Initialize scaler including scaler type and min/max values."""
        assert scale_min < scale_max

        # Set min/max range for scaler
        self.type = scale_type
        self.range = ScaleRange(
            min_value=jnp.array(scale_min),
            max_value=jnp.array(scale_max),
        )

        # Statistical parameters
        self.dimension: int = 0
        self.params = ScalerParams()

        self.number_of_warnings: int = 0
        self.max_number_of_warnings: Optional[int] = None

        # Set scaler type function
        self._transform = getattr(self, f"{self.type}")

    def fit(self, data: Array) -> None:
        """
        Fit descriptor scaler parameters using the input data.
        Bach-wise sampling is also possible (see `this`_ for more details).

        .. _this: https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        """
        data = jnp.atleast_2d(data)  # type: ignore

        if self.params.nsamples == 0:
            self.dimension = data.shape[1]
            self.params = _init_scaler_params_from(data)
        else:
            if data.shape[1] != self.dimension:
                logger.error(
                    f"Data dimension doesn't match: {data.shape[1]} (expected {self.dimension})",
                    exception=ValueError,
                )
            self.params = _fit_scaler(self.params, data)

    def __call__(self, array: Array, warnings: bool = False) -> Array:
        """
        Transform the input descriptor values based on the scaler parameters.

        This method has to be called after fitting scaler over the dataset,
        or statistical parameters are already loaded (e.g. saved file).
        """
        if warnings:
            self._check_warnings(array)
        return self._transform(array)

    def set_max_number_of_warnings(self, number: Optional[int] = None) -> None:
        """Set the maximum number of warning for out of range descriptor values."""
        self.max_number_of_warnings = number
        self.number_of_warnings = 0
        logger.debug(
            f"Setting the maximum number of scaler warnings: {self.max_number_of_warnings}"
        )

    def _check_warnings(self, array: Array) -> None:
        """
        Check whether the output scaler values exceed the predefined min/max range values or not.

        If it's the case, it keeps counting the number of warnings and
        raises an error when it exceeds the maximum number.

        An out of range descriptor value is in fact an indication of
        the descriptor extrapolation which has to be avoided.
        """
        if self.max_number_of_warnings is None:
            return

        self.number_of_warnings += int(_get_number_of_warnings(self.params, array))

        if self.number_of_warnings >= self.max_number_of_warnings:
            logger.warning(
                "Exceeding maximum number scaler warnings (extrapolation warning): "
                f"{self.number_of_warnings} (max={self.max_number_of_warnings})"
            )

    def center(self, array: Array) -> Array:
        return _center(self.params, array)

    def scale(self, array: Array) -> Array:
        return _scale(self.range, self.params, array)

    def scale_center(self, array: Array) -> Array:
        return _scale_center(self.range, self.params, array)

    def scale_center_sigma(self, array: Array) -> Array:
        return _scale_center_sigma(self.range, self.params, array)

    def save(self, filename: Path) -> None:
        """Save scaler parameters into file."""
        logger.debug(f"Saving scaler parameters into '{str(filename)}'")
        with open(str(filename), "w") as file:
            file.write(f"{'# Min':<23s} {'Max':<23s} {'Mean':<23s} {'Sigma':<23s}\n")
            for i in range(self.dimension):
                file.write(
                    f"{self.params.minval[i]:<23.15E}"
                    f"{self.params.maxval[i]:<23.15E}"
                    f"{self.params.mean[i]:<23.15E}"
                    f"{self.params.sigma[i]:<23.15E}\n"
                )

    def load(self, filename: Path, dtype: Optional[Dtype] = None) -> None:
        """Load scaler parameters from file."""
        logger.debug(f"Loading scaler parameters from '{str(filename)}'")
        data = np.loadtxt(str(filename)).T
        dtype = dtype if dtype is not None else default_dtype.FLOATX
        self.dimension = data.shape[1]
        self.params = ScalerParams(
            nsamples=_to_jax_int(1),
            mean=jnp.array(data[2], dtype=dtype),
            sigma=jnp.array(data[3], dtype=dtype),
            minval=jnp.array(data[0], dtype=dtype),
            maxval=jnp.array(data[1], dtype=dtype),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(scale_type='{self.type}', "
            f"scale_range=({int(self.range.min_value)}, {int(self.range.max_value)}))"
        )

    def __bool__(self) -> bool:
        return False if len(self.params.mean) == 0 else True


@jax.jit
def _init_scaler_params_from(data: Array) -> ScalerParams:
    return ScalerParams(
        nsamples=_to_jax_int(data.shape[0]),
        mean=jnp.mean(data, axis=0),
        sigma=jnp.std(data, axis=0),
        maxval=jnp.max(data, axis=0),
        minval=jnp.min(data, axis=0),
    )


@jax.jit
def _fit_scaler(
    params: ScalerParams,
    data: Array,
) -> ScalerParams:
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

    maxval = jnp.maximum(params.maxval, new_max)
    minval = jnp.minimum(params.minval, new_min)
    nsamples = params.nsamples + n
    return ScalerParams(nsamples, mean, sigma, minval, maxval)


@jax.jit
def _center(
    params: ScalerParams,
    array: Array,
) -> Array:
    return array - params.mean


@jax.jit
def _scale(
    scale_range: ScaleRange,
    params: ScalerParams,
    array: Array,
) -> Array:
    return scale_range.min_value + (scale_range.max_value - scale_range.min_value) * (
        array - params.minval
    ) / (params.maxval - params.minval)


@jax.jit
def _scale_center(
    scale_range: ScaleRange,
    params: ScalerParams,
    array: Array,
) -> Array:
    return scale_range.min_value + (scale_range.max_value - scale_range.min_value) * (
        array - params.mean
    ) / (params.maxval - params.minval)


@jax.jit
def _scale_center_sigma(
    scale_range: ScaleRange,
    params: ScalerParams,
    array: Array,
) -> Array:
    return (
        scale_range.min_value
        + (scale_range.min_value - scale_range.max_value)
        * (array - params.mean)
        / params.sigma
    )


@jax.jit
def _get_number_of_warnings(
    params: ScalerParams,
    array: Array,
) -> Array:
    if array.ndim == 2:
        gt = jax.lax.gt(array, params.maxval[None, :])
        lt = jax.lax.gt(params.minval[None, :], array)
    else:
        gt = jax.lax.gt(array, params.maxval)
        lt = jax.lax.gt(params.minval, array)
    # alternative counting is using sum
    return jnp.any(jnp.logical_or(gt, lt))
