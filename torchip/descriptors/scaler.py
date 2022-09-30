
from multiprocessing.connection import wait
from ..logger import logger
from ..config import dtype as _dtype
from ..config import device as _device
from .base import BaseTorchipClass
from torch import Tensor
from pathlib import Path
import torch
import numpy as np


class DescriptorScaler(BaseTorchipClass):
  """
  Scale descriptor values.
  TODO: see https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
  TODO: add warnings for out-of-distribution samples
  """
  def __init__(self, 
      scale_type: str = 'scale_center', 
      scale_min: float = 0.0,
      scale_max: float = 1.0,
      dtype: torch.dtype = None,
      device: torch.device = None,
      max_number_of_warnings = 100,
      ) -> None:
    """
    Initialize scaler including scaler type and min/max values.
    """
    # Set min/max range for scaler
    self.scale_type = scale_type
    self.scale_min = scale_min
    self.scale_max = scale_max
    self.dtype = dtype if dtype else _dtype.FLOAT
    self.device = device if device else _device.DEVICE
    logger.debug(f"Initializing {self}")

    # Statistical parameters
    self.nsamples = 0       # number of samples
    self.dimension = None   # dimension of each sample
    self.mean = None        # mean array of all fitted descriptor values
    self.sigma = None       # standard deviation
    self.min =  None        # minimum
    self.max =  None        # maximum

    self.number_of_warnings = 0
    self.max_number_of_warnings = max_number_of_warnings

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
      self.sigma = torch.std(data, dim=0, unbiased=False)
      self.max = torch.max(data, dim=0)[0]
      self.min = torch.min(data, dim=0)[0]
    else:
      # Check data dimension
      if data.shape[1] != self.dimension:
        logger.error(f"Data dimension doesn't match previous observation ({self.dim}): {data.shape[0]}",
                      exception=ValueError)

      # New data (batch)
      new_mean = torch.mean(data, dim=0)
      new_sigma = torch.std(data, dim=0, unbiased=False)
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

  def __call__(self, x: Tensor, warnings: bool = False) -> Tensor:
    """
    Transform the input descriptor values base on the selected scaler type.
    This merhod has to be called when fit method is called ``batch-wise`` over all descriptor values, 
    or statistical parameters are read from a saved file. 

    Args:
        x (Tensor): input 
        warnings (bool): input 

    Returns:
        Tensor: scaled input
    """    
    scaled_x = self._transform(x)
    if warnings:
      self._check_warnings(scaled_x)
    
    return scaled_x 

  @torch.no_grad()
  def _check_warnings(self, x: Tensor) -> None:
    """
    Check whether the output scaler values exceed the predefined min/max range values or not.
    if so, it keeps counting the number of warnings and raises an error if it exceeds the maximum number.
    out of range descriptor values is an indication of descriptor extrapolation which has to be avoided.  

    :param val: scaled values of descriptor
    :type val: Tensor
    """    
    gt = torch.gt(x, self.max).detach()
    lt = torch.gt(self.min, x).detach()
    self.number_of_warnings += int(torch.sum(torch.logical_or(gt, lt)))

    if self.number_of_warnings >= self.max_number_of_warnings:
      logger.error(f"Exceeded the maximum number of scaler warnings ({self.max_number_of_warnings})", 
                   exception=ValueError)

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
    kwargs = {"dtype": self.dtype, "device":self.device}
    self.min   = torch.tensor(data[:, 0], **kwargs) 
    self.max   = torch.tensor(data[:, 1], **kwargs)
    self.mean  = torch.tensor(data[:, 2], **kwargs)
    self.sigma = torch.tensor(data[:, 3], **kwargs)

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}(scale_type='{self.scale_type}', scale_min={self.scale_min}, scale_max={self.scale_max})"
