from collections import defaultdict
from mlap.descriptors.asf.angular import AngularSymmetryFunction
from ...logger import logger
from ...structure import Structure
from ..base import Descriptor
from .cutoff_function import CutoffFunction
from .radial import G1, G2, RadialSymmetryFunction
from typing import Union
import torch

dtype = torch.double
device = torch.device("cpu")


class ASF (Descriptor):
  """
  Atomic Symmetry Function (ASF) descriptor.
  ASF is vector of different radial and angular terms.
  # TODO: ASF should be independent of the structure 
  """

  def __init__(self, element: str) -> None:
    self.element = element
    self._radial = []
    self._angular = []
    # TODO: read from input.nn

  def add(self, asf: Union[RadialSymmetryFunction,  AngularSymmetryFunction],
                neighbor_element1: str, 
                neighbor_element2: str = None) -> None:
    """
    This method adds an input radial symmetry function to the list of ASFs.
    # TODO: tuple of dict? (tuple is fine if it's used internally)
    """
    if isinstance(asf, RadialSymmetryFunction):
      self._radial.append((asf, self.element, neighbor_element1))
    elif isinstance(asf, RadialSymmetryFunction):
      self._angular((asf, self.element, neighbor_element1, neighbor_element2))
    else:
      msg = f"Unknown input symmetry function object type"
      logger.error(msg)
      raise TypeError(msg)

  def __call__(self, structure:Structure, aid: int) -> torch.tensor: 
    """
    Calculate descriptor values for the input given structure.
    """
    x = structure.position
    at = structure.atom_type
    nn  = structure.neighbor.numbers
    ngb = structure.neighbor.indices
    emap= structure.element_map

    result = torch.zeros(len(self._radial), dtype=dtype, device=device)

    # Get the list of neighboring atom indices
    ni_ = ngb[aid, :nn[aid]]
    # Calculate the distances of neighboring atoms and the corresponding atom types
    # TODO: apply PBC
    rij = torch.norm(x[ni_]-x[0], dim=1) 
    tij = at[ni_] 
    # Loop of radial terms
    for i, g in enumerate(self._radial):
      # Find the neighboring atom indices that match the given ASF cutoff radius and atom type
      ngb_rc_ = (rij < g[0].r_cutoff ).detach()
      ngb_ = torch.nonzero(torch.logical_and(ngb_rc_, tij == emap[g[2]]), as_tuple=True)[0]
      # Apply the ASF term and sum over the neighboring atoms
      result[i] = torch.sum( g[0].kernel(rij[ngb_] ), dim=0)

    return result


