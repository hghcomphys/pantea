from typing import Mapping, Optional, Type

import jax.numpy as jnp
from jaxip.base import _Base
from jaxip.logger import logger
from jaxip.types import Array


class ErrorMetric(_Base):
    """
    A base error metric class.
    Note: gradient calculations is disabled for all error metrics.
    """

    def __init__(self) -> None:
        def mse(*, prediction: Array, target: Array) -> Array:
            return ((target - prediction) ** 2).mean()

        self._mse_metric = mse

    def __call__(self, prediction: Array, target: Array, factor=None) -> Array:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MSE(ErrorMetric):
    """
    Mean squared error metric
    """

    def __call__(
        self, prediction: Array, target: Array, factor: Optional[float] = None
    ) -> Array:
        return self._mse_metric(prediction=prediction, target=target)


class RMSE(MSE):
    """
    Root mean squared error metric
    """

    def __call__(
        self, prediction: Array, target: Array, factor: Optional[float] = None
    ) -> Array:
        return jnp.sqrt(self._mse_metric(prediction=prediction, target=target))


class MSEpa(MSE):
    """
    Mean squared error per atom metric.
    MSE of energy per atom
    MSE of force
    """

    def __call__(self, prediction: Array, target: Array, factor: float = 1.0) -> Array:
        return self._mse_metric(prediction=prediction, target=target) / factor


class RMSEpa(RMSE):
    """
    Root mean squared error per atom metric.
    RMSE of energy per atom
    RMSE of force
    """

    def __call__(self, prediction: Array, target: Array, factor: int = 1) -> Array:
        return jnp.sqrt(self._mse_metric(prediction=prediction, target=target)) / factor


def init_error_metric(metric_type: str, **kwargs) -> Optional[MSE]:
    """
    An utility function to create a given type of error metric.

    :param metric_type: MSE, RMSE, MSEpa, EMSEpa
    :type metric_type: str
    :return: An instance of desired error metric
    :rtype: ErrorMetric
    """
    _map_error_metric: Mapping[str, Type[ErrorMetric]] = {
        "MSE": MSE,
        "RMSE": RMSE,
        "MSEpa": MSEpa,
        "RMSEpa": RMSEpa,
        # Any new defined metrics must be added here.
    }
    try:
        return _map_error_metric[metric_type](**kwargs)
    except KeyError:
        logger.error(f"Unknown error metric '{metric_type}'", exception=KeyError)
