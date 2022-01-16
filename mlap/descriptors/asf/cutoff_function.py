from ...logger import logger
import math
import torch


class CutoffFunction:
  """
  This class contains different cutoff function used for ASF descriptor.
  TODO: optimization
  TODO: add logger
  """
  def __init__(self, r_cutoff: float, fn_type: str = "tanh"):
    self.r_cutoff = r_cutoff
    self.fn_type = fn_type.lower()
    # Set cutoff function
    try:
      self.fn = getattr(self, f"_{self.fn_type}")
    except AttributeError:
      msg = f"'{self.__class__.__name__}' has no implemented cutoff function '{self.fn_type}'"
      logger.error(msg)
      raise NotImplementedError(msg)

    # logger.info(f"Initialize {self.__class__.__name__} with r_cutoff({self.r_cutoff}) and function({self.fn_type})")

  def __call__(self, r: torch.Tensor) -> torch.Tensor:
    return torch.where( r < self.r_cutoff, self.fn(r), torch.zeros_like(r))

  def _tanh(self, r: torch.Tensor) -> torch.Tensor:
    return torch.tanh(1.0 - r/self.r_cutoff).pow(3)

  def _cos(self, r: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.cos(math.pi * r/self.r_cutoff) + 1.0)

  def _exp(self, r: torch.Tensor) -> torch.Tensor:
    return torch.exp(1.0 - 1.0 / (1.0 - (r/self.r_cutoff)**2) )
