from collections import defaultdict
from ...logger import logger
from ...structure import Structure
from ..base import Descriptor
from .cutoff_function import CutoffFunction
import torch

dtype = torch.double
device = torch.device("cpu")


class ASF (Descriptor):
  """
  Atomic Symmetry Function (ASF) descriptor.
  ASF is vector of different radial and angular terms.
  """

  def __init__(self):
    self._terms = {}

  def __call__(self, structure:Structure, aid: int) -> torch.tensor: 
    """
    Calculate descriptor values for the input given structure.
    """
    x = structure.position
    nn  = structure.neighbor.numbers
    ngb = structure.neighbor.indices
    self.r_cutoff = structure.neighbor.r_cutoff  # has to be set durning class instantiation
    self._cutoff_func= CutoffFunction(self.r_cutoff, "tanh")

    # Loop over neighbors
    # TODO: optimize
    result = torch.tensor(0.0, dtype=torch.float)
    rij = torch.norm(x[ngb[aid, :nn[aid]]]-x[0], dim=1)
    neighbors_ = torch.nonzero( rij < self.r_cutoff, as_tuple=True)[0]
    for nb in neighbors_.detach().tolist():
        result = result + self.kernel(rij[nb])
    return result

  def kernel(self, rij: torch.tensor) -> torch.tensor:
    # TODO: improve design for kernel arguments
    return torch.exp(-rij**2) * self._cutoff_func(rij) #  torch.tanh(1.0 - rij/self.r_cutoff).pow(3)



