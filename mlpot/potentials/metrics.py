from ..logger import logger
from ..base import _Base
from torch import Tensor
from typing import Mapping
import torch


class ErrorMetric(_Base):
    """
    A base error metric class.
    Note: gradient calculations is disabled for all error metrics.
    """

    def __init__(self, **kwargs):
        self._mse_metric = torch.nn.MSELoss(**kwargs)

    @torch.no_grad()
    def __call__(self, prediction: Tensor, target: Tensor, factor=None) -> Tensor:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MSE(ErrorMetric):
    """
    Mean squared error metric
    """

    @torch.no_grad()
    def __call__(self, prediction: Tensor, target: Tensor, factor=None) -> Tensor:
        return self._mse_metric(prediction, target)


class RMSE(MSE):
    """
    Root mean squared error metric
    """

    @torch.no_grad()
    def __call__(self, prediction: Tensor, target: Tensor, factor=None) -> Tensor:
        return torch.sqrt(self._mse_metric(prediction, target))


class MSEpa(MSE):
    """
    Mean squared error per atom metric.
    MSE of energy per atom
    MSE of force
    """

    @torch.no_grad()
    def __call__(self, prediction: Tensor, target: Tensor, factor: int = 1) -> Tensor:
        return self._mse_metric(prediction, target) / factor


class RMSEpa(RMSE):
    """
    Root mean squared error per atom metric.
    RMSE of energy per atom
    RMSE of force
    """

    @torch.no_grad()
    def __call__(self, prediction: Tensor, target: Tensor, factor: int = 1) -> Tensor:
        return torch.sqrt(self._mse_metric(prediction, target)) / factor


def create_error_metric(metric_type: str, **kwargs) -> ErrorMetric:
    """
    An utility function to create a given type of error metric.

    :param metric_type: MSE, RMSE, MSEpa, EMSEpa
    :type metric_type: str
    :return: An instance of desired error metric
    :rtype: ErrorMetric
    """
    _map_error_metric: Mapping[str, ErrorMetric] = {
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
