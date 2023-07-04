from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from jaxip.logger import logger
from jaxip.types import Array, Dtype
from jaxip.types import _dtype


class Scaler:
    """
    Scale descriptor values.

    Scaling parameters are calculated by fitting over the samples in the dataset.
    Available scaler information are as follows:

    * minimum
    * maximum
    * mean
    * sigma (standard deviation)

    Scaler is also used to warn when setting out-of-distribution samples base
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
        self.scale_min: float = scale_min
        self.scale_max: float = scale_max

        # Statistical parameters
        self.nsamples: int = 0  # number of samples
        self.dimension: int = 0  # dimension of each sample
        self.mean: Array = jnp.asarray([])  # mean array of all fitted descriptor values
        self.sigma: Array = jnp.asarray([])  # standard deviation
        self.min: Array = jnp.asarray([])  # minimum
        self.max: Array = jnp.asarray([])  # maximum

        self.number_of_warnings: int = 0
        self.max_number_of_warnings: Optional[int] = None

        logger.debug(f"Initializing {self}")

        # Set scaler type function
        self._transform = getattr(self, f"{self.scale_type}")

    def fit(self, data: Array) -> None:
        """
        Fit scaler parameters using the given input data.
        Bach-wise sampling is also possible (see `this`_ for more details).

        .. _this: https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        """
        data = jnp.atleast_2d(data)  # type: ignore

        if self.nsamples == 0:
            self.nsamples = data.shape[0]
            self.dimension = data.shape[1]
            self.mean = jnp.mean(data, axis=0)
            self.sigma = jnp.std(data, axis=0)
            self.max = jnp.max(data, axis=0)
            self.min = jnp.min(data, axis=0)
        else:
            if data.shape[1] != self.dimension:
                logger.error(
                    f"Data dimension doesn't match: {data.shape[1]} (expected {self.dimension})",
                    exception=ValueError,
                )

            # New data (batch)
            new_mean: Array = jnp.mean(data, axis=0)
            new_sigma: Array = jnp.std(data, axis=0)
            new_min: Array = jnp.min(data, axis=0)
            new_max: Array = jnp.max(data, axis=0)
            m, n = self.nsamples, data.shape[0]

            # Calculate quantities for entire data
            mean = self.mean  # immutable
            self.mean = (
                m / (m + n) * mean + n / (m + n) * new_mean
            )  # self.mean is now a new array and different from the above mean variable
            self.sigma = jnp.sqrt(
                m / (m + n) * self.sigma**2
                + n / (m + n) * new_sigma**2
                + m * n / (m + n) ** 2 * (mean - new_mean) ** 2
            )
            self.max = jnp.maximum(self.max, new_max)
            self.min = jnp.minimum(self.min, new_min)
            self.nsamples += n

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
            gt = jax.lax.gt(array, self.max[None, :])
            lt = jax.lax.gt(self.min[None, :], array)
        else:
            gt = jax.lax.gt(array, self.max)
            lt = jax.lax.gt(self.min, array)

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
            array - self.min
        ) / (self.max - self.min)

    def scale_center(self, array: Array) -> Array:
        return self.scale_min + (self.scale_max - self.scale_min) * (
            array - self.mean
        ) / (self.max - self.min)

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
                    f"{self.min[i]:<23.15E} {self.max[i]:<23.15E} {self.mean[i]:<23.15E} {self.sigma[i]:<23.15E}\n"
                )

    def load(self, filename: Path, dtype: Optional[Dtype] = None) -> None:
        """Load scaler parameters from file."""
        logger.debug(f"Loading scaler parameters from '{str(filename)}'")
        data = np.loadtxt(str(filename)).T
        dtype = dtype if dtype is not None else _dtype.FLOATX
        self.nsamples = 1
        self.dimension = data.shape[1]
        self.min = jnp.asarray(data[0], dtype=dtype)
        self.max = jnp.asarray(data[1], dtype=dtype)
        self.mean = jnp.asarray(data[2], dtype=dtype)
        self.sigma = jnp.asarray(data[3], dtype=dtype)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(scale_type='{self.scale_type}', "
            f"scale_min={self.scale_min}, scale_max={self.scale_max})"
        )

    def __bool__(self) -> bool:
        return False if len(self.mean) == 0 else True
