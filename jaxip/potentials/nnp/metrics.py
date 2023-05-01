from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Callable, Mapping, Optional, Type

import jax.numpy as jnp

from jaxip.logger import logger
from jaxip.types import Array


class ErrorMetric(metaclass=ABCMeta):
    """A base error metric class."""

    def __init__(self) -> None:
        """Initialize base error metric."""

        def mse(*, prediction: Array, target: Array) -> Array:
            return ((target - prediction) ** 2).mean()

        self._mse_metric: Callable[..., Array] = mse

    @classmethod
    def create(cls, metric_type: str) -> ErrorMetric:
        """
        Create the given type of error metric.

        :param metric_type: MSE, RMSE, MSEpa, EMSEpa
        :return: An instance of desired error metric
        """
        _map_error_metric: Mapping[str, Type[ErrorMetric]] = {
            "MSE": MSE,
            "RMSE": RMSE,
            "MSEpa": MSEpa,
            "RMSEpa": RMSEpa,
            # Any new defined metrics must be added here.
        }
        try:
            error_metric = _map_error_metric[metric_type]()
        except KeyError:
            logger.error(
                f"Unknown error metric '{metric_type}'",
                exception=KeyError,
            )
        return error_metric  # type: ignore

    @abstractmethod
    def __call__(self, prediction: Array, target: Array, factor=None) -> Array:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MSE(ErrorMetric):
    """Mean squared error."""

    def __call__(
        self, prediction: Array, target: Array, factor: Optional[float] = None
    ) -> Array:
        return self._mse_metric(prediction=prediction, target=target)


class RMSE(MSE):
    """Root mean squared error."""

    def __call__(
        self, prediction: Array, target: Array, factor: Optional[float] = None
    ) -> Array:
        return jnp.sqrt(self._mse_metric(prediction=prediction, target=target))


class MSEpa(MSE):
    """
    Mean squared error per atom.
    MSE of energy **per atom**
    MSE of force
    """

    def __call__(self, prediction: Array, target: Array, factor: float = 1.0) -> Array:
        return self._mse_metric(prediction=prediction, target=target) / factor


class RMSEpa(RMSE):
    """
    Root mean squared error per atom.
    RMSE of energy **per atom**
    RMSE of force
    """

    def __call__(self, prediction: Array, target: Array, factor: float = 1.0) -> Array:
        return jnp.sqrt(self._mse_metric(prediction=prediction, target=target)) / factor
