from ...logger import logger
import math
import torch


class CutoffFunction:
  """
  This class contains different cutoff function used for ASF descriptor.
  TODO: optimization
  TODO: add logger
  TODO: add poly 3 & 4 funcions
  See N2P2 -> https://compphysvienna.github.io/n2p2/api/cutoff_functions.html?highlight=cutoff#
              https://compphysvienna.github.io/n2p2/topics/keywords.html?highlight=cutoff_type
  TODO: define inv_r_cutoff
  """
  _TANH_PRE = math.pow((math.e + 1 / math.e) / (math.e - 1 / math.e), 3)

  def __init__(self, r_cutoff: float, cutoff_type: str = "tanh"):
    self.r_cutoff = r_cutoff
    self.cutoff_type = cutoff_type.lower()
    self.inv_r_cutoff = 1.0 / self.r_cutoff
    # Set cutoff type function
    try:
      self.fn = getattr(self, f"_{self.cutoff_type}")
    except AttributeError:
      msg = f"'{self.__class__.__name__}' has no implemented cutoff function '{self.cutoff_type}'"
      logger.error(msg)
      raise NotImplementedError(msg)

  def __call__(self, r: torch.Tensor) -> torch.Tensor:
    return torch.where( r < self.r_cutoff, self.fn(r), torch.zeros_like(r))

  def _hard(self, r: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(r)

  def _tanhu(self, r: torch.Tensor) -> torch.Tensor:
    return torch.tanh(1.0 - r * self.inv_r_cutoff).pow(3)
  
  def _tanh(self, r: torch.Tensor) -> torch.Tensor:
    return self._TANH_PRE * torch.tanh(1.0 - r * self.inv_r_cutoff).pow(3)

  def _cos(self, r: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.cos(math.pi * r * self.inv_r_cutoff) + 1.0)

  def _exp(self, r: torch.Tensor) -> torch.Tensor:
    return torch.exp(1.0 - 1.0 / (1.0 - (r * self.inv_r_cutoff)**2) )

  def _poly1(self, r: torch.Tensor) -> torch.Tensor:
    return (2.0*r - 3.0) * r**2 + 1.0

  def _poly2(self, r: torch.Tensor) -> torch.Tensor:
    return ((15.0 - 6.0*r) * r - 10) * r**3 + 1.0

  def __repr__(self) -> str:
      return f"{self.__class__.__name__}(r_cutoff={self.r_cutoff}, cutoff_type='{self.cutoff_type}')"
