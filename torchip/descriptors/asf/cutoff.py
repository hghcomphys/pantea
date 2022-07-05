from ...logger import logger
from torch import Tensor
import math
import torch
import cutoff_cpp


class CutoffFunction:
  # """
  # This class contains different cutoff function used for ASF descriptor.
  # TODO: optimization
  # TODO: add logger
  # TODO: add poly 3 & 4 funcions
  # See N2P2 -> https://compphysvienna.github.io/n2p2/api/cutoff_functions.html?highlight=cutoff#
  #             https://compphysvienna.github.io/n2p2/topics/keywords.html?highlight=cutoff_type
  # TODO: define inv_r_cutoff
  # """
  _TANH_PRE = math.pow((math.e + 1 / math.e) / (math.e - 1 / math.e), 3)

  def __init__(self, r_cutoff: float, cutoff_type: str = "tanh"):
    self.r_cutoff = r_cutoff
    self.cutoff_type = cutoff_type.lower()
    self.inv_r_cutoff = 1.0 / self.r_cutoff
    # Set cutoff type function
    try:
      self.cfn = getattr(self, f"{self.cutoff_type}")
    except AttributeError:
      logger.error(f"'{self.__class__.__name__}' has no cutoff function '{self.cutoff_type}'",
                    exception=NotImplementedError)

  def __call__(self, r: Tensor) -> Tensor:
    # TODO: add C++ kernel
    return torch.where( r < self.r_cutoff, self.cfn(r), torch.zeros_like(r))

  def hard(self, r: Tensor) -> Tensor:
    # return torch.ones_like(r)
    return cutoff_cpp._hard(r)

  def tanhu(self, r: Tensor) -> Tensor:
    # return torch.tanh(1.0 - r * self.inv_r_cutoff).pow(3)
    return cutoff_cpp._tanhu(r, self.inv_r_cutoff)
  
  def tanh(self, r: Tensor) -> Tensor:
    # return self._TANH_PRE * torch.tanh(1.0 - r * self.inv_r_cutoff).pow(3)
    return cutoff_cpp._tanh(r, self.inv_r_cutoff)

  def cos(self, r: Tensor) -> Tensor:
    # return 0.5 * (torch.cos(math.pi * r * self.inv_r_cutoff) + 1.0)
    return cutoff_cpp._cos(r, self.inv_r_cutoff)

  def exp(self, r: Tensor) -> Tensor:
    # return torch.exp(1.0 - 1.0 / (1.0 - (r * self.inv_r_cutoff)**2) )
    return cutoff_cpp._exp(r, self.inv_r_cutoff)

  def poly1(self, r: Tensor) -> Tensor:
    # return (2.0*r - 3.0) * r**2 + 1.0
    return cutoff_cpp._poly1(r)

  def poly2(self, r: Tensor) -> Tensor:
    # return ((15.0 - 6.0*r) * r - 10) * r**3 + 1.0
    return cutoff_cpp._poly2(r)

  def __repr__(self) -> str:
      return f"{self.__class__.__name__}(r_cutoff={self.r_cutoff}, cutoff_type='{self.cutoff_type}')"
