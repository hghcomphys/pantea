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

  def __init__(self, element: str= None):
    self.element = element
    self._terms = {}

  def __call__(self, structure:Structure, aid: int) -> torch.tensor: 
    """
    Calculate descriptor values for the input given structure.
    """
    self.r_cutoff = structure.neighbor.r_cutoff  # has to be set durning class instantiation
    self.cutoff_function= CutoffFunction(self.r_cutoff, "tanh")

    x = structure.position
    at = structure.atom_type
    nn  = structure.neighbor.numbers
    ngb = structure.neighbor.indices

    # Get the list of neighboring atom indices
    ni_ = ngb[aid, :nn[aid]]
    # Calculate the distances of neighboring atoms and the corresponding atom types
    rij = torch.norm(x[ni_]-x[0], dim=1)
    tij = at[ni_] 
    # Find the neighboring atom indices which match the given ASF cutoff and atom type
    ngb_rc_ = (rij < self.r_cutoff).detach()
    ngb_ = torch.nonzero(torch.logical_and(ngb_rc_, tij == 1), as_tuple=True)[0]
    print(tij)
    # Apply the ASF term and sum over neighboring atoms
    result = torch.sum(self.kernel(rij[ngb_]), dim=0)

    return result

  def kernel(self, rij: torch.tensor) -> torch.tensor:
    # TODO: improve design for kernel arguments
    return torch.exp(-rij**2) * self.cutoff_function(rij) #  torch.tanh(1.0 - rij/self.r_cutoff).pow(3)



