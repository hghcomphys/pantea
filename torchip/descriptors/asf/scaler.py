
from ...logger import logger
from ...config import CFG
from ...structure import Structure
from ...utils.batch import create_batch
from ..base import Descriptor
from typing import Dict, Union, List
from pathlib import Path
import torch
import numpy as np


def std_(data: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
  """
  A utility function which is defined because of the difference observed when using torch.std function.
  This occurs for torch (numpy version is fine).
  """
  return torch.sqrt(torch.mean((data - mean)**2, dim=0))


class AtomicSymmetryFunctionScaler:
  """
  Scale descriptor values.
  TODO: see https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
  TODO: add warnings for out-of-distribution samples
  """
  def __init__(self, 
      descriptor: Descriptor,
      scale_type: str = 'scale_center', 
      scale_min: float = 0.0,
      scale_max: float = 1.0,
      ) -> None:
    """
    Initialize scaler including scaler type and min/max values.
    """
    self.descriptor = descriptor
    # Set min/max range for scaler
    self.scale_type = scale_type
    self.scale_min = scale_min
    self.scale_max = scale_max
    logger.debug(repr(self))

    # Statistical parameters
    self.sample = 0         # number of samples
    self.dimension = None   # dimension of each sample
    self.mean = None        # mean array of all fitted descriptor values
    self.sigma = None       # standard deviation
    self.min =  None        # minimum
    self.max =  None        # maximum

    # Set scaler type function     
    self._transform = getattr(self, f'_{self.scale_type}')   

  def __repr__(self) -> str:
      return f"{self.__class__.__name__}(scale_type='{self.scale_type}', scale_min={self.scale_min}, scale_max={self.scale_max})"     

  def fit(self, structure: Structure, aid: Union[List[int], int] = None):  
    """
    Fit scaler parameters of the descriptor based on the given input structure and atom ids. 
    The input argument should be the same as descriptor's call method.
    """
    logger.info("Fitting descriptor scaler")
    descriptor_values = self.descriptor(structure, aid)
    self._fit(descriptor_values)

  def _fit(self, descriptor_values: torch.Tensor) -> None:
    """
    This method extracts stattical quantities from the input tensor of descriptor values.
    """
    data = descriptor_values.detach()  # no gradient history is required
    data = torch.atleast_2d(data)

    # First time initialization
    if self.sample == 0:
      self.sample = data.shape[0]
      self.dimension = data.shape[1]
      self.mean = torch.mean(data, dim=0)
      self.sigma = std_(data, self.mean) #torch.std(data, dim=0)
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
      new_sigma = std_(data, new_mean) #torch.std(data, dim=0)
      new_min = torch.min(data, dim=0)[0]
      new_max = torch.max(data, dim=0)[0]
      m, n = float(self.sample), data.shape[0]

      # Calculate quantities for entire data
      mean = self.mean 
      self.mean = m/(m+n)*mean + n/(m+n)*new_mean  # self.mean is now a new array and different from the above mean variable
      self.sigma  = torch.sqrt( m/(m+n)*self.sigma**2 + n/(m+n)*new_sigma**2 + m*n/(m+n)**2 * (mean - new_mean)**2 ) 
      self.max = torch.maximum(self.max, new_max)
      self.min = torch.minimum(self.min, new_min)
      self.sample += n

  def __call__(self, descriptor_values: Dict[str, torch.Tensor]):
    """
    Transform the input descriptor values base on the selected scaler type.
    This merhod has to be called when fit method is called batch-wise over all descriptor values, 
    or statistical parameters are read from a saved file. 
    """
    return self._transform(descriptor_values)

  def _center(self, G: torch.Tensor) -> torch.Tensor:
    return G - self.mean

  def _scale(self, G: torch.Tensor) -> torch.Tensor:
    return self.scale_min + (self.scale_max - self.scale_min) * (G - self.min) / (self.max - self.min)

  def _scale_center(self, G: torch.Tensor) -> torch.Tensor:
    return self.scale_min + (self.scale_max - self.scale_min) * (G - self.mean) / (self.max - self.min)
  
  def _scale_sigma(self, G: torch.Tensor) -> torch.Tensor:
    return self.scale_min + (self.scale_max - self.scale_min) * (G - self.mean) / self.sigma

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
    self.sample = 1
    self.dimension = data.shape[1]
    self.min   = torch.tensor(data[:, 0], device=CFG["device"]) # TODO: dtype?
    self.max   = torch.tensor(data[:, 1], device=CFG["device"])
    self.mean  = torch.tensor(data[:, 2], device=CFG["device"])
    self.sigma = torch.tensor(data[:, 3], device=CFG["device"])


# Define ASF scaler alias
AsfScaler = AtomicSymmetryFunctionScaler