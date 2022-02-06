
from .cutoff import CutoffFunction
import math
import torch

class AngularSymmetryFunction:
  """
  Three body symmetry function.
  TODO: add other variant of angular symmetry functions.
  """
  def __init__(self, r_cutoff: float, cutoff_type: str):
    self.r_cutoff = r_cutoff
    self.cutoff_type = cutoff_type
    self.cutoff_function = CutoffFunction(r_cutoff, cutoff_type)

  def kernel(self, rij: torch.tensor) -> torch.tensor:
    raise NotImplementedError


class G3(AngularSymmetryFunction):
  """
  Angular symmetry function.
  """
  def __init__(self, r_cutoff: float, cutoff_type: str, eta: float, zeta: float, lambda0: float, r_shift: float) -> None:
    super().__init__(r_cutoff, cutoff_type)
    self.eta = eta
    self.zeta = zeta
    self.lambda0 = lambda0
    self.r_shift = r_shift
    self._scale_factor = math.pow(2.0, 1.0-self.zeta)
    
  def kernel(self, rij: torch.tensor, rik: torch.tensor, rjk: torch.tensor, cost: torch.tensor) -> torch.tensor:
    res = self._scale_factor * torch.pow(1 + self.lambda0 * cost, self.zeta) * torch.exp( -self.eta * (rij**2 + rik**2 + rjk**2) )  # TODO: r_shift
    return res * self.cutoff_function(rij) * self.cutoff_function(rik) * self.cutoff_function(rjk)


class G9(AngularSymmetryFunction):
  """
  Modified angular symmetry function.
  Ref -> J. Behler, J. Chem. Phys. 134, 074106 (2011)
  """
  def __init__(self, r_cutoff: float, cutoff_type: str, eta: float, zeta: float, lambda0: float, r_shift: float) -> None:
    super().__init__(r_cutoff, cutoff_type)
    self.eta = eta
    self.zeta = zeta
    self.lambda0 = lambda0
    self.r_shift = r_shift
    self._scale_factor = math.pow(2.0, 1.0-self.zeta)
    
  def kernel(self, rij: torch.tensor, rik: torch.tensor, rjk: torch.tensor, cost: torch.tensor) -> torch.tensor:
    res = self._scale_factor * torch.pow(1 + self.lambda0 * cost, self.zeta) * torch.exp( -self.eta * (rij**2 + rik**2) ) # TODO: r_shift
    return res * self.cutoff_function(rij) * self.cutoff_function(rik)
