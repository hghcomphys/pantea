from ...logger import logger
from ...structure import Structure
from ..base import Descriptor
from .angular import AngularSymmetryFunction
from .radial import G1, G2, RadialSymmetryFunction
from typing import Union
import torch

dtype = torch.double
device = torch.device("cpu")


class ASF(Descriptor):
  """
  Atomic Symmetry Function (ASF) descriptor.
  ASF is a vector of different radial and angular terms.
  # TODO: ASF should be independent of the input structure 
  """
  def __init__(self, element: str) -> None:
    self.element = element
    self._radial = []
    self._angular = []
    # TODO: read from input.nn

  def add(self, symmetry_function: Union[RadialSymmetryFunction,  AngularSymmetryFunction],
                neighbor_element1: str, 
                neighbor_element2: str = None) -> None:
    """
    This method adds an input radial symmetry function to the list of ASFs.
    # TODO: tuple of dict? (tuple is fine if it's used internally)
    # TODO: solve the confusion for aid, starting from 0 or 1?!
    """
    if isinstance(symmetry_function, RadialSymmetryFunction):
      self._radial.append((symmetry_function, self.element, neighbor_element1))
    elif isinstance(symmetry_function, RadialSymmetryFunction):
      self._angular((symmetry_function, self.element, neighbor_element1, neighbor_element2))
    else:
      msg = f"Unknown input symmetry function type"
      logger.error(msg)
      raise TypeError(msg)

  def __call__(self, structure:Structure, aid: int) -> torch.tensor: 
    """
    Calculate descriptor values for the input given structure.
    """
    x = structure.position
    at = structure.atom_type
    nn  = structure.neighbor_number
    ngb = structure.neighbor_index
    emap= structure.element_map

    result = torch.zeros(len(self._radial), dtype=dtype, device=device)

    # Check aid atom type
    if not emap[self.element] == at[aid]:
      msg = f"Inconsistent central element ('{self.element}'): input aid={aid} ('{emap[int(at[aid])]}')"
      logger.error(msg)
      raise AssertionError(msg)

    # Get the list of neighboring atom indices
    ni_ = ngb[aid, :nn[aid]]
    # Calculate the distances of neighboring atoms and the corresponding atom types
    # TODO: apply PBC
    rij = torch.norm(x[ni_]-x[0], dim=1) 
    tij = at[ni_] 
    # Loop of radial terms
    for i, sf in enumerate(self._radial):
      # Find the neighboring atom indices that match the given ASF cutoff radius and atom type
      ngb_rc_ = (rij < sf[0].r_cutoff ).detach()
      ngb_ = torch.nonzero(torch.logical_and(ngb_rc_, tij == emap[sf[2]]), as_tuple=True)[0]
      # Apply the ASF term and sum over the neighboring atoms
      result[i] = torch.sum( sf[0].kernel(rij[ngb_] ), dim=0)

    return result


