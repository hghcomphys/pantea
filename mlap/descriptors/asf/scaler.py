
from ...logger import logger
from typing import Dict
from collections import defaultdict
import numpy as np
import torch


class AtomicSymmetryFunctionScaler:
  """
  Scale descriptor values.
  TODO: see https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
  """
  def __init__(self, **kwargs):
    self.num = 0                          # number of samples
    self.dim = None                       # dimension of samples
    self.mean = None                      # mean array of all fitted descriptor values
    self.sigma = None                     # standard deviation
    self.max = None                       # maximum
    self.min = None                       # minimum

    self.smin = kwargs.get("smin", 0.0)
    self.smax = kwargs.get("smax", 1.0)

  
  def fit(self, descriptor_values: torch.Tensor) -> None:
    """
    This method extract stattical quantities from the input descriptor values.
    This method can also extract the required quantities even batch-wise.
    """
    data = descriptor_values.detach()  # no gradient history is required
    data = torch.atleast_2d(data)

    # First time initialization
    if self.num == 0:
      self.num = data.shape[0]
      self.dim = data.shape[1]
      self.mean = torch.mean(data, dim=0)
      self.sigma = torch.std(data, dim=0)
      self.max = torch.max(data, dim=0)
      self.min = torch.min(data, dim=0)
    else:
      # Check data dimension
      if data.shape[1] != self.dim:
        msg = f"Data dimension doesn't match previous observation ({self.dim}): {data.shape[0]}"
        logger.error(msg)
        raise ValueError(msg)

      # New data (batch)
      new_mean = torch.mean(data, dim=0)
      new_sigma  = torch.std(data, dim=0)
      new_min = torch.min(data, dim=0)
      new_max = torch.max(data, dim=0)
      m, n = float(self.num), data.shape[0]
      mean = self.mean

      self.mean = m/(m+n)*mean + n/(m+n)*new_mean
      self.sigma  = torch.sqrt( m/(m+n)*self.std**2 + n/(m+n)*new_sigma**2 + m*n/(m+n)**2 * (mean - new_mean)**2 )
      self.min = torch.minimum(self.min, new_min)
      self.max = torch.maximum(self.max, new_max)
      self.num += n

   
  def transform(self, descriptor_values: Dict[str, torch.Tensor]):
    pass

  def _center(self, G: torch.Tensor, element: str) -> torch.Tensor:
    return G - self.mean

  def _scale(self, G: torch.Tensor, element: str) -> torch.Tensor:
    return self.smin + (self.smax - self.smin) * (G - self.min) / (self.max - self.min)

  def _scalecenter(self, G: torch.Tensor, element: str) -> torch.Tensor:
    return self.smin + (self.smax - self.smin) * (G - self.mean) / (self.max- self.min)
  
  def _scalesigma(self, G: torch.Tensor, element: str) -> torch.Tensor:
    return self.smin + (self.smax - self.smin) * (G - self.mean) / self.sigma


AsfScaler = AtomicSymmetryFunctionScaler