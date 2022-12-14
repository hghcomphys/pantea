from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from mlpot.base import _Base
from mlpot.logger import logger
from mlpot.types import Array
from mlpot.types import dtype as _dtype


class DescriptorScaler(_Base):
    """
    Scale descriptor values.
    TODO: see https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
    TODO: add warnings for out-of-distribution samples
    """

    def __init__(
        self,
        scale_type: str = "scale_center",
        scale_min: float = 0.0,
        scale_max: float = 1.0,
        dtype: Optional[jnp.dtype] = _dtype.FLOATX,
    ) -> None:
        """
        Initialize scaler including scaler type and min/max values.
        """
        # Set min/max range for scaler
        self.scale_type = scale_type
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.dtype = dtype
        logger.debug(f"Initializing {self}")

        # Statistical parameters
        self.nsamples: int = 0  # number of samples
        self.dimension: int = None  # dimension of each sample
        self.mean: Array = None  # mean array of all fitted descriptor values
        self.sigma: Array = None  # standard deviation
        self.min: Array = None  # minimum
        self.max: Array = None  # maximum

        self.number_of_warnings: int = 0
        self.max_number_of_warnings: int = None

        # Set scaler type function
        self._transform = getattr(self, f"_{self.scale_type}")

    def fit(self, data: Array) -> None:
        """
        This method fits the scaler parameters based on the given input tensor.
        It also works also in a batch-wise form.
        """
        data = jnp.atleast_2d(data)

        if self.nsamples == 0:
            self.nsamples = data.shape[0]
            self.dimension = data.shape[1]
            self.mean = jnp.mean(data, axis=0)
            self.sigma = jnp.std(data, axis=0)
            self.max = jnp.max(data, axis=0)
            self.min = jnp.min(data, axis=0)
        else:
            # Check data dimension
            if data.shape[1] != self.dimension:
                logger.error(
                    f"Data dimension doesn't match: {data.shape[1]} (expected {self.dimension})",
                    exception=ValueError,
                )

            # New data (batch)
            new_mean = jnp.mean(data, axis=0)
            new_sigma = jnp.std(data, axis=0)
            new_min = jnp.min(data, axis=0)
            new_max = jnp.max(data, axis=0)
            m, n = float(self.nsamples), data.shape[0]

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

    def __call__(self, x: Array, warnings: bool = False) -> Array:
        """
        Transform the input descriptor values base on the selected scaler type.
        This method has to be called when fit method was already applied batch-wise over all descriptor values,
        or statistical parameters are read from a saved file.
        """

        if warnings:
            self._check_warnings(x)

        return self._transform(x)

    def set_max_number_of_warnings(self, number: int) -> None:
        self.max_number_of_warnings = number
        self.number_of_warnings = 0
        logger.debug(
            f"Setting the maximum number of scaler warnings: {self.max_number_of_warnings}"
        )

    def _check_warnings(self, x: Array) -> None:
        """
        Check whether the output scaler values exceed the predefined min/max range values or not.
        if so, it keeps counting the number of warnings and raises an error if it exceeds the maximum number.
        An out of range descriptor value is in fact an indication of
        the descriptor extrapolation which has to be avoided.
        """
        if self.max_number_of_warnings is None:
            return

        gt = jax.lax.gt(x, self.max)
        lt = jax.lax.gt(self.min, x)

        self.number_of_warnings += int(
            jnp.any(jnp.logical_or(gt, lt))
        )  # alternative counting is using sum

        if self.number_of_warnings >= self.max_number_of_warnings:
            logger.warning(
                "Exceeding maximum number scaler warnings (extrapolation warning): "
                f"{self.number_of_warnings} (max={self.max_number_of_warnings})"
            )

    def _center(self, x: Array) -> Array:
        return x - self.mean

    def _scale(self, x: Array) -> Array:
        return self.scale_min + (self.scale_max - self.scale_min) * (x - self.min) / (
            self.max - self.min
        )

    def _scale_center(self, x: Array) -> Array:
        return self.scale_min + (self.scale_max - self.scale_min) * (x - self.mean) / (
            self.max - self.min
        )

    def _scale_center_sigma(self, x: Array) -> Array:
        return (
            self.scale_min
            + (self.scale_max - self.scale_min) * (x - self.mean) / self.sigma
        )

    def save(self, filename: Path) -> None:
        """
        Save scaler parameters into a file.
        """
        with open(str(filename), "w") as file:
            file.write(f"{'# Min':<23s} {'Max':<23s} {'Mean':<23s} {'Sigma':<23s}\n")
            for i in range(self.dimension):
                file.write(
                    f"{self.min[i]:<23.15E} {self.max[i]:<23.15E} {self.mean[i]:<23.15E} {self.sigma[i]:<23.15E}\n"
                )

    def load(self, filename: Path) -> None:
        """
        Load scaler parameters from the file.
        """
        data = np.loadtxt(str(filename)).T
        self.nsamples = 1
        self.dimension = data.shape[1]

        self.min = jnp.asarray(data[0], dtype=self.dtype)
        self.max = jnp.asarray(data[1], dtype=self.dtype)
        self.mean = jnp.asarray(data[2], dtype=self.dtype)
        self.sigma = jnp.asarray(data[3], dtype=self.dtype)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(scale_type='{self.scale_type}',"
            f"scale_min={self.scale_min}, scale_max={self.scale_max})"
        )
