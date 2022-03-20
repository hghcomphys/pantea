
from ...logger import logger
from .cutoff import CutoffFunction
from .symmetry import SymmetryFunction
import math
import torch
import angular_cpp

class AngularSymmetryFunction(SymmetryFunction):
  """
  Three body symmetry function.
  TODO: add other variant of angular symmetry functions (see N2P2 documentation).
  """
  def kernel(self, rij: torch.Tensor, rik: torch.Tensor, rjk: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


class G3(AngularSymmetryFunction):
  """
  Angular symmetry function.
  """
  def __init__(self, cutoff_function: CutoffFunction, eta: float, zeta: float, lambda0: float, r_shift: float) -> None:
    self.eta = eta
    self.zeta = zeta
    self.lambda0 = lambda0
    self.r_shift = r_shift
    self._scale_factor = math.pow(2.0, 1.0-self.zeta)
    self._params = [self.eta, self.zeta, self.lambda0, self.r_shift, self._scale_factor]
    super().__init__(cutoff_function)
    
  def kernel(self, rij: torch.Tensor, rik: torch.Tensor, rjk: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
    # res = self._scale_factor * torch.pow(1 + self.lambda0 * cost, self.zeta) * torch.exp( -self.eta * (rij**2 + rik**2 + rjk**2) )  # TODO: r_shift
    res = angular_cpp.g3_kernel(rij, rik, rjk, cost, self._params)
    return res * self.cutoff_function(rij) * self.cutoff_function(rik) * self.cutoff_function(rjk)


class G9(G3): # AngularSymmetryFunction
  """
  Modified angular symmetry function.
  Ref -> J. Behler, J. Chem. Phys. 134, 074106 (2011)
  """
  # def __init__(self, cutoff_function: CutoffFunction, eta: float, zeta: float, lambda0: float, r_shift: float) -> None:
  #   self.eta = eta
  #   self.zeta = zeta
  #   self.lambda0 = lambda0
  #   self.r_shift = r_shift
  #   self._scale_factor = math.pow(2.0, 1.0-self.zeta)
  #   self._params = [self.eta, self.zeta, self.lambda0, self.r_shift, self._scale_factor]
  #   super().__init__(cutoff_function)
    
  def kernel(self, rij: torch.Tensor, rik: torch.Tensor, rjk: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
    #res = self._scale_factor * torch.pow(1 + self.lambda0 * cost, self.zeta) * torch.exp( -self.eta * (rij**2 + rik**2) ) # TODO: r_shift
    res = angular_cpp.g9_kernel(rij, rik, rjk, cost, self._params)
    return res * self.cutoff_function(rij) * self.cutoff_function(rik)
