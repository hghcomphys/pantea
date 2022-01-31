from mlap.descriptors.base import Descriptor
from mlap.structure import Structure
from .cutoff import CutoffFunction
import torch


class RadialSymmetryFunction:
  """
  Two body symmetry function.
  TODO: define generic **params input arguments in the base class?
  TODO: add __call__() method?
  TODO: define a internal cutoff radius
  TODO: add other variant of radial symmetry functions.
  TODO: add logging when initializing each symmetry function.
  """
  def __init__(self, r_cutoff: float, cutoff_type: str):
    self.r_cutoff = r_cutoff
    self.cutoff_type = cutoff_type
    self.cutoff_function = CutoffFunction(r_cutoff, cutoff_type)

  def kernel(self, rij: torch.tensor) -> torch.tensor:
    raise NotImplementedError


class G1(RadialSymmetryFunction):
  """
  Plain cutoff function.
  """
  def __init__(self, r_cutoff: float, cutoff_type: str):
    super().__init__(r_cutoff, cutoff_type)

  def kernel(self, rij: torch.tensor) -> torch.tensor:
    return self.cutoff_function(rij)


class G2(RadialSymmetryFunction):
  """
  Radial exponential term.
  """
  def __init__(self, r_cutoff: float, cutoff_type: str, r_shift: float, eta: float):
    super().__init__(r_cutoff, cutoff_type)
    self.r_shift = r_shift
    self.eta = eta

  def kernel(self, rij: torch.tensor) -> torch.tensor:
    return torch.exp( -self.eta * (rij - self.r_shift)**2 ) * self.cutoff_function(rij)