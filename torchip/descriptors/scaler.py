
from ..logger import logger
from ..config import CFG
from torch import Tensor
from typing import Dict
from pathlib import Path
import torch
import numpy as np


# def std_(data: Tensor, mean: Tensor) -> Tensor:
#   """
#   An utility function which is defined because of the difference observed when using torch.std function.
#   This occurs for torch (numpy version is fine).
#   """
#   return torch.sqrt(torch.mean((data - mean)**2, dim=0))


class DescriptorScaler:
  """
  Scale descriptor values.
  TODO: see https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
  TODO: add warnings for out-of-distribution samples
  """
  def __init__(self, 
      scale_type: str = 'scale_center', 
      scale_min: float = 0.0,
      scale_max: float = 1.0,
      ) -> None:
    """
    Initialize scaler including scaler type and min/max values.
    """
    # Set min/max range for scaler
    self.scale_type = scale_type
    self.scale_min = scale_min
    self.scale_max = scale_max
    logger.debug(f"{self.__class__.__name__}(scale_type='{self.scale_type}', scale_min={self.scale_min}, scale_max={self.scale_max})")

    # Statistical parameters
    self.nsamples = 0       # number of samples
    self.dimension = None   # dimension of each sample
    self.mean = None        # mean array of all fitted descriptor values
    self.sigma = None       # standard deviation
    self.min =  None        # minimum
    self.max =  None        # maximum

    # Set scaler type function     
    self._transform = getattr(self, f'_{self.scale_type}')    

  def fit(self, x: Tensor) -> None:
    """
    This method fits the scaler parameters based on the given input tensor.
    It also works also in a batch-wise form.
    """
    data = x.detach()  # no gradient history is required
    data = torch.atleast_2d(data)

    # First time initialization
    if self.nsamples == 0:
      self.nsamples = data.shape[0]
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
      m, n = float(self.nsamples), data.shape[0]

      # Calculate quantities for entire data
      mean = self.mean.clone()
      self.mean = m/(m+n)*mean + n/(m+n)*new_mean  # self.mean is now a new array and different from the above mean variable
      self.sigma  = torch.sqrt( m/(m+n)*self.sigma**2 + n/(m+n)*new_sigma**2 + m*n/(m+n)**2 * (mean - new_mean)**2 ) 
      self.max = torch.maximum(self.max, new_max)
      self.min = torch.minimum(self.min, new_min)
      self.nsamples += n

  def __call__(self, x: Tensor) -> Tensor:
    """
    Transform the input descriptor values base on the selected scaler type.
    This merhod has to be called when fit method is called ``batch-wise`` over all descriptor values, 
    or statistical parameters are read from a saved file. 

    Args:
        x (Tensor): input 

    Returns:
        Tensor: scaled input
    """    
    return self._transform(x)

  def _center(self, x: Tensor) -> Tensor:
    """
    Subtract the mean value from the input tensor.
    """    
    return x - self.mean

  def _scale(self, x: Tensor) -> Tensor:
    return self.scale_min + (self.scale_max - self.scale_min) * (x - self.min) / (self.max - self.min)

  def _scale_center(self, x: Tensor) -> Tensor:
    return self.scale_min + (self.scale_max - self.scale_min) * (x - self.mean) / (self.max - self.min)
  
  def _scale_center_sigma(self, x: Tensor) -> Tensor:
    return self.scale_min + (self.scale_max - self.scale_min) * (x - self.mean) / self.sigma

  def save(self, filename: Path) -> None:
    """
    Save scaler parameters into file.
    """
    with open(str(filename), "w") as file:
      file.write(f"{'# Min':<23s} {'Max':<23s} {'Mean':<23s} {'Sigma':<23s}\n")   
      for i in range(self.dimension):
        file.write(f"{self.min[i]:<23.15E} {self.max[i]:<23.15E} {self.mean[i]:<23.15E} {self.sigma[i]:<23.15E}\n")

  def load(self, filename: Path) -> None:
    """
    Load scaler parameters from file.
    """
    data = np.loadtxt(str(filename))
    self.nsamples = 1
    self.dimension = data.shape[1]
    kwargs = { "dtype": CFG["dtype"], "device":CFG["device"] }
    self.min   = torch.tensor(data[:, 0], **kwargs) 
    self.max   = torch.tensor(data[:, 1], **kwargs)
    self.mean  = torch.tensor(data[:, 2], **kwargs)
    self.sigma = torch.tensor(data[:, 3], **kwargs)
