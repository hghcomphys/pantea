from ...logger import logger
from ...base import BaseTorchipClass
from .cutoff import CutoffFunction
from torch import Tensor


class SymmetryFunction(BaseTorchipClass):
  """
  A base class for symmetry functions. 
  All symmetry functions (i.e. radial and angular) must derive from this base class.
  """
  def __init__(self, cfn: CutoffFunction):
    self.cfn = cfn
    logger.debug(repr(self))

  def kernel(self, *args, **kwargs) -> Tensor:
    raise NotImplementedError

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}(" + \
      ', '.join(['{}={!r}'.format(k, v) for k, v in self.__dict__.items() if "_"]) + ")"

  @property
  def r_cutoff(self) -> float:
    return self.cfn.r_cutoff