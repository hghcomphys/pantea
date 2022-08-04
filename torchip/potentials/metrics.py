import torch
from torch import Tensor


class ErrorMetric:
  """
  A base error metric class.
  Note: gradient calculations is disabled for all error metrics.
  """
  def __init__(self, **kwargs):
    self._mse_metric = torch.nn.MSELoss(**kwargs)
  
  @torch.no_grad()
  def __call__(self, prediction: Tensor, target: Tensor, factor=None) -> Tensor:
    raise NotImplementedError()

  def __str__(self) -> str:
    return self.__class__.__name__


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
    return self._mse_metric(prediction, target)/factor


class RMSEpa(RMSE):
  """
  Root mean squared error per atom metric.
  RMSE of energy per atom
  RMSE of force
  """
  @torch.no_grad()
  def __call__(self, prediction: Tensor, target: Tensor, factor: int = 1) -> Tensor:
    return torch.sqrt(self._mse_metric(prediction, target))/factor