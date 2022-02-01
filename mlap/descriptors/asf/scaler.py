
from typing import Dict
from collections import defaultdict
import numpy as np
import torch


class AtomicSymmetryFunctionScaler:
  """
  Scale descriptor values.
  TODO: see https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
  """
  def __init__(self, element: str, **kwargs):
    self.min = np.inf
    self.max = -np.inf
    self.sum = 0.0
    self.num = 0
    self.smin = kwargs.get("smin", 0.0)
    self.smax = kwargs.get("smax", 1.0)
  
  def fit(self, descriptor_values: torch.Tensor) -> None:
    """
    This method extract stattical quantities from the input descriptor values.
    """

    for element in descriptor_values:

      val = descriptor_values[element].detach()

      min_ = torch.min(val).numpy()
      if min_ < self.min[element]:
        self.min[element] = min_

      max_ = torch.max(val).numpy()
      if max_ > self.max[element]:
        self.max[element] = max_

      self.sum[element] += torch.sum(val).numpy()
      self.num[element] += val.shape[0]

   
  def transform(self, descriptor_values: Dict[str, torch.Tensor]):
    pass

  def _center(self, G: torch.Tensor, element: str) -> torch.Tensor:
      mean_ = self.sum[element] / self.num[element]
      return G - mean_

  def _scale(self, G: torch.Tensor, element: str) -> torch.Tensor:
      return self.smin + (self.smax - self.smin) * (G - self.min[element]) / (self.max[element]- self.min[element])

  def _scalecenter(self, G: torch.Tensor, element: str) -> torch.Tensor:
      mean_ = self.sum[element] / self.num[element]
      return self.smin + (self.smax - self.smin) * (G - mean_) / (self.max[element]- self.min[element])
  
  def _scalesigma(self, G: torch.Tensor, element: str) -> torch.Tensor:
      mean_ = self.sum[element] / self.num[element]
      sigma_ = None # TODO
      return self.smin + (self.smax - self.smin) * (G - mean_) / sigma_


