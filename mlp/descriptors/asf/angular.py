
from ...logger import logger
from .cutoff import CutoffFunction
import math
import torch


class AngularSymmetryFunction:
  """
  Three body symmetry function.
  TODO: add other variant of angular symmetry functions.
  """
  def __init__(self, cutoff_function: CutoffFunction):
    self.cutoff_function = cutoff_function
    logger.debug(repr(self))

  def kernel(self, rij: torch.tensor) -> torch.tensor:
    raise NotImplementedError

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}(" + \
      ', '.join(['{}={!r}'.format(k, v) for k, v in self.__dict__.items() if "_"]) + ")"

  @property
  def r_cutoff(self) -> float:
    return self.cutoff_function.r_cutoff


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
    super().__init__(cutoff_function)
    
  def kernel(self, rij: torch.tensor, rik: torch.tensor, rjk: torch.tensor, cost: torch.tensor) -> torch.tensor:
    res = self._scale_factor * torch.pow(1 + self.lambda0 * cost, self.zeta) * torch.exp( -self.eta * (rij**2 + rik**2 + rjk**2) )  # TODO: r_shift
    return res * self.cutoff_function(rij) * self.cutoff_function(rik) * self.cutoff_function(rjk)


class G9(AngularSymmetryFunction):
  """
  Modified angular symmetry function.
  Ref -> J. Behler, J. Chem. Phys. 134, 074106 (2011)
  """
  def __init__(self, cutoff_function: CutoffFunction, eta: float, zeta: float, lambda0: float, r_shift: float) -> None:
    self.eta = eta
    self.zeta = zeta
    self.lambda0 = lambda0
    self.r_shift = r_shift
    self._scale_factor = math.pow(2.0, 1.0-self.zeta)
    super().__init__(cutoff_function)
    
  def kernel(self, rij: torch.tensor, rik: torch.tensor, rjk: torch.tensor, cost: torch.tensor) -> torch.tensor:
    res = self._scale_factor * torch.pow(1 + self.lambda0 * cost, self.zeta) * torch.exp( -self.eta * (rij**2 + rik**2) ) # TODO: r_shift
    return res * self.cutoff_function(rij) * self.cutoff_function(rik)
