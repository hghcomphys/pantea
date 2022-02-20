
from ...logger import logger
from typing import Dict
from collections import defaultdict
import torch


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
      scaler_type: str = 'scale_center', 
      scaler_min: float = 0.0,
      scaler_max: float = 1.0,
      ) -> None:
    """
    Initialize scaler including scaler type and min/max values.
    """
    # Set min/max range for scaler
    self.scaler_type =  scaler_type
    self.scaler_min = scaler_min
    self.scaler_max = scaler_max

    # Statistical parameters
    self.sample = None      # number of samples
    self.dimension = None   # dimension of each sample
    self.mean = None        # mean array of all fitted descriptor values
    self.sigma = None       # standard deviation
    self.min =  None        # minimum
    self.max =  None        # maximum

    # Set scaler type function     
    self._scaler_function = getattr(self, f'_{self.scaler_type}')         

  def fit(self, descriptor_values: torch.Tensor) -> None:
    """
    This method extract stattical quantities from the input descriptor values.
    This method can also extract the required quantities even batch-wise.
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
    return self._scaler_function(descriptor_values)

  def _center(self, G: torch.Tensor) -> torch.Tensor:
    return G - self.mean

  def _scale(self, G: torch.Tensor) -> torch.Tensor:
    return self.scaler_min + (self.scaler_max - self.scaler_min) * (G - self.min) / (self.max - self.min)

  def _scale_center(self, G: torch.Tensor) -> torch.Tensor:
    return self.scaler_min + (self.scaler_max - self.scaler_min) * (G - self.mean) / (self.max- self.min)
  
  def _scale_sigma(self, G: torch.Tensor) -> torch.Tensor:
    return self.scaler_min + (self.scaler_max - self.scaler_min) * (G - self.mean) / self.sigma


# Define ASF scaler alias
AsfScaler = AtomicSymmetryFunctionScaler