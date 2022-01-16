from collections import defaultdict
from ...logger import logger
from ...structure import Structure
from ..base import Descriptor
from .cutoff_function import CutoffFunction
from .radial import G1, G2
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
    self.cutoff_type = self.cutoff_function.fn_type

    x = structure.position
    at = structure.atom_type
    nn  = structure.neighbor.numbers
    ngb = structure.neighbor.indices
    
    self._terms["G1"] = G1(self.r_cutoff, self.cutoff_type)
    self._terms["G2"] = G2(self.r_cutoff, self.cutoff_type, r_shift=0.0, eta=0.3)
    result = torch.zeros(len(self._terms), dtype=dtype, device=device)

    # Get the list of neighboring atom indices
    ni_ = ngb[aid, :nn[aid]]
    # Calculate the distances of neighboring atoms and the corresponding atom types
    rij = torch.norm(x[ni_]-x[0], dim=1)
    tij = at[ni_] 
    # Find the neighboring atom indices which match the given ASF cutoff and atom type
    ngb_rc_ = (rij < self.r_cutoff).detach()
    ngb_ = torch.nonzero(torch.logical_and(ngb_rc_, tij == 1), as_tuple=True)[0]
    # Apply the ASF term and sum over neighboring atoms
    for i, descriptor in enumerate(self._terms.values()):
      result[i] = torch.sum( descriptor.kernel(rij[ngb_] ), dim=0)

    return result


