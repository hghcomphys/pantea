from ...logger import logger
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
  def __init__(self, cutoff_function: CutoffFunction):
    self.cutoff_function = cutoff_function
    logger.debug(repr(self))

  def kernel(self, rij: torch.tensor) -> torch.tensor:
    raise NotImplementedError

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}(" + ', '.join(['{}={!r}'.format(k, v) for k, v in self.__dict__.items()]) + ")"

  @property
  def r_cutoff(self) -> float:
    return self.cutoff_function.r_cutoff


class G1(RadialSymmetryFunction):
  """
  Plain cutoff function.
  """
  def __init__(self, cutoff_function: CutoffFunction):
    super().__init__(cutoff_function)

  def kernel(self, rij: torch.tensor) -> torch.tensor:
    return self.cutoff_function(rij)


class G2(RadialSymmetryFunction):
  """
  Radial exponential term.
  """
  def __init__(self, cutoff_function: CutoffFunction, r_shift: float, eta: float):
    self.r_shift = r_shift
    self.eta = eta
    super().__init__(cutoff_function)

  def kernel(self, rij: torch.tensor) -> torch.tensor:
    return torch.exp( -self.eta * (rij - self.r_shift)**2 ) * self.cutoff_function(rij)