from ...logger import logger
from .symmetry import SymmetryFunction
from .cutoff import CutoffFunction
import torch
import radial_cpp


class RadialSymmetryFunction(SymmetryFunction):
  """
  Two body symmetry function.
  TODO: define generic **params input arguments in the base class?
  TODO: add __call__() method?
  TODO: define a internal cutoff radius
  TODO: add other variant of radial symmetry functions.
  TODO: add logging when initializing each symmetry function.
  """
  def kernel(self, rij: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


class G1(RadialSymmetryFunction):
  """
  Plain cutoff function.
  """
  def __init__(self, cutoff_function: CutoffFunction):
    super().__init__(cutoff_function)

  def kernel(self, rij: torch.Tensor) -> torch.Tensor:
    # No cpp kernel is required
    return self.cutoff_function(rij) 


class G2(RadialSymmetryFunction):
  """
  Radial exponential term.
  """
  def __init__(self, cutoff_function: CutoffFunction, r_shift: float, eta: float):
    self.r_shift = r_shift
    self.eta = eta
    self._params = [self.eta, self.r_shift]
    super().__init__(cutoff_function)

  def kernel(self, rij: torch.Tensor) -> torch.Tensor:
    # return torch.exp( -self.eta * (rij - self.r_shift)**2 ) * self.cutoff_function(rij)
    return radial_cpp.g2_kernel(rij, self._params) * self.cutoff_function(rij)