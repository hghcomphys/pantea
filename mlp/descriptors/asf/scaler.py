
from ...logger import logger
from typing import Dict
from collections import defaultdict
import numpy as np
import torch


class AtomicSymmetryFunctionScaler:
  """
  Scale descriptor values.
  TODO: see https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
  TODO: add warnings for out-of-distribution samples
  """
  def __init__(self, **kwargs):
    # Statistical parameters
    self.sample = kwargs.get("sample", 0)       # number of samples
    self.dimension = kwargs.get("dimension")    # dimension of each sample
    self.mean = kwargs.get("mean")              # mean array of all fitted descriptor values
    self.sigma = kwargs.get("sigma")            # standard deviation
    self.min =  kwargs.get("min")               # minimum
    self.max =  kwargs.get("max")               # maximum
    # Set min/max range for scaler
    self.smin = kwargs.get("scale_min_short", 0.0)                 
    self.smax = kwargs.get("scale_max_short", 1.0)     
    self.scaler_type =  kwargs.get('scaler_type', 'scale_center')    
    # Set scaler type function     
    self._scaler_function = getattr(self, f'_{self.scaler_type}')         

  def fit(self, descriptor_values: torch.Tensor) -> None:
    """
    This method extract stattical quantities from the input descriptor values.
    This method can also extract the required quantities even batch-wise.
    """
    data = descriptor_values.detach()  # no gradient history is required
    data = torch.atleast_2d(data) #torch.unsqueeze(data, dim=0) if data.ndim <2 else data

    # First time initialization
    if self.sample == 0:
      self.sample = data.shape[0]
      self.dimension = data.shape[1]
      self.mean = torch.mean(data, dim=0)
      self.sigma = torch.std(data, dim=0)
      self.max = torch.max(data, dim=0)[0]
      self.min = torch.min(data, dim=0)[0]
    else:
      # Check data dimension
      if data.shape[1] != self.dimension:
        msg = f"Data dimension doesn't match previous observation ({self.dim}): {data.shape[0]}"
        logger.error(msg)
        raise ValueError(msg)

      # New data (batch)
      new_mean = torch.mean(data, dim=0)
      new_sigma = torch.std(data, dim=0)
      new_min = torch.min(data, dim=0)[0]
      new_max = torch.max(data, dim=0)[0]
      m, n = float(self.sample), data.shape[0]

      # Calculate quantities for entire data
      mean = self.mean 
      self.mean = m/(m+n)*mean + n/(m+n)*new_mean  # self.mean is now a new array
      self.sigma  = torch.sqrt( m/(m+n)*self.sigma**2 + n/(m+n)*new_sigma**2 + m*n/(m+n)**2 * (mean - new_mean)**2 ) # TODO: unbiased std?
      self.max = torch.maximum(self.max, new_max)
      self.min = torch.minimum(self.min, new_min)
      self.sample += n

  def transform(self, descriptor_values: Dict[str, torch.Tensor]):
    """
    Transform the input descriptor values base on the selected scaler type.
    This merhod has to be called when fit method is called batch-wise over all descriptor values, 
    or statistical parameters are read from a saved file. 
    """
    return self._scaler_function(descriptor_values)

  def _center(self, G: torch.Tensor) -> torch.Tensor:
    return G - self.mean

  def _scale(self, G: torch.Tensor) -> torch.Tensor:
    return self.smin + (self.smax - self.smin) * (G - self.min) / (self.max - self.min)

  def _scale_center(self, G: torch.Tensor) -> torch.Tensor:
    return self.smin + (self.smax - self.smin) * (G - self.mean) / (self.max- self.min)
  
  def _scale_sigma(self, G: torch.Tensor) -> torch.Tensor:
    return self.smin + (self.smax - self.smin) * (G - self.mean) / self.sigma


AsfScaler = AtomicSymmetryFunctionScaler