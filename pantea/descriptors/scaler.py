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
    """Scaler statistical parameters"""

    nsamples: Array = _to_jax_int(0)
    mean: Array = jnp.array([])
    sigma: Array = jnp.array([])
    minval: Array = jnp.array([])
    maxval: Array = jnp.array([])


@jax.jit
def _init_params_from(data: Array) -> ScalerParams:
    return ScalerParams(
        nsamples=_to_jax_int(data.shape[0]),
        mean=jnp.mean(data, axis=0),
        sigma=jnp.std(data, axis=0),
        maxval=jnp.max(data, axis=0),
        minval=jnp.min(data, axis=0),
    )


@jax.jit
def _fit_scaler(params: ScalerParams, data: Array) -> ScalerParams:
    # Calculate statistical parameters for a new batch of data
    new_mean: Array = jnp.mean(data, axis=0)
    new_sigma: Array = jnp.std(data, axis=0)
    new_min: Array = jnp.min(data, axis=0)
    new_max: Array = jnp.max(data, axis=0)
    m, n = params.nsamples, data.shape[0]
    # Calculate scaler new params for the entire data
    mean = m / (m + n) * params.mean + n / (m + n) * new_mean
    sigma = jnp.sqrt(
        m / (m + n) * params.sigma**2
        + n / (m + n) * new_sigma**2
        + m * n / (m + n) ** 2 * (params.mean - new_mean) ** 2
    )
    maxval = jnp.maximum(params.maxval, new_max)
    minval = jnp.minimum(params.minval, new_min)
    nsamples = params.nsamples + n
    return ScalerParams(nsamples, mean, sigma, minval, maxval)


class DescriptorScaler:
    """
    Scale descriptor values.

    Scaling parameters are calculated by fitting over the samples in the dataset.
    Available scaler information are as follows:

    * mean
    * sigma (standard deviation)
    * maxval
    * minval

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
        self.scale_type: str = scale_type
        self.scale_min: Array = jnp.array(scale_min)
        self.scale_max: Array = jnp.array(scale_max)

        # Statistical parameters
        self.dimension: int = 0
        self.params = ScalerParams()

        self.number_of_warnings: int = 0
        self.max_number_of_warnings: Optional[int] = None

        # Set scaler type function
        self._transform = getattr(self, f"{self.scale_type}")

    def fit(self, data: Array) -> None:
        """
        Fit scaler parameters using the given input data.
        Bach-wise sampling is also possible (see `this`_ for more details).

        .. _this: https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        """
        data = jnp.atleast_2d(data)  # type: ignore

        if self.params.nsamples == 0:
            self.dimension = data.shape[1]
            self.params = _init_params_from(data)
        else:
            if data.shape[1] != self.dimension:
                logger.error(
                    f"Data dimension doesn't match: {data.shape[1]} (expected {self.dimension})",
                    exception=ValueError,
                )
            self.params = _fit_scaler(self.params, data)

    def __call__(self, array: Array, warnings: bool = False) -> Array:
        """
        Transform the input descriptor values base on the scaler parameters.

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

        gt: Array
        lt: Array
        if array.ndim == 2:
            gt = jax.lax.gt(array, self.maxval[None, :])
            lt = jax.lax.gt(self.minval[None, :], array)
        else:
            gt = jax.lax.gt(array, self.maxval)
            lt = jax.lax.gt(self.minval, array)

        self.number_of_warnings += int(
            jnp.any(jnp.logical_or(gt, lt))
        )  # alternative counting is using sum

        if self.number_of_warnings >= self.max_number_of_warnings:
            logger.warning(
                "Exceeding maximum number scaler warnings (extrapolation warning): "
                f"{self.number_of_warnings} (max={self.max_number_of_warnings})"
            )

    def center(self, array: Array) -> Array:
        return array - self.mean

    def scale(self, array: Array) -> Array:
        return self.scale_min + (self.scale_max - self.scale_min) * (
            array - self.minval
        ) / (self.maxval - self.minval)

    def scale_center(self, array: Array) -> Array:
        return self.scale_min + (self.scale_max - self.scale_min) * (
            array - self.mean
        ) / (self.maxval - self.minval)

    def scale_center_sigma(self, array: Array) -> Array:
        return (
            self.scale_min
            + (self.scale_max - self.scale_min) * (array - self.mean) / self.sigma
        )

    def save(self, filename: Path) -> None:
        """Save scaler parameters into file."""
        logger.debug(f"Saving scaler parameters into '{str(filename)}'")
        with open(str(filename), "w") as file:
            file.write(f"{'# Min':<23s} {'Max':<23s} {'Mean':<23s} {'Sigma':<23s}\n")
            for i in range(self.dimension):
                file.write(
                    f"{self.minval[i]:<23.15E} {self.maxval[i]:<23.15E} {self.mean[i]:<23.15E} {self.sigma[i]:<23.15E}\n"
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
            f"{self.__class__.__name__}(scale_type='{self.scale_type}', "
            f"scale_min={self.scale_min}, scale_max={self.scale_max})"
        )

    def __bool__(self) -> bool:
        return False if len(self.mean) == 0 else True

    @property
    def mean(self) -> Array:
        return self.params.mean

    @property
    def sigma(self) -> Array:
        return self.params.sigma

    @property
    def minval(self) -> Array:
        return self.params.minval

    @property
    def maxval(self) -> Array:
        return self.params.maxval

    @property
    def nsamples(self) -> int:
        return self.params.nsamples
