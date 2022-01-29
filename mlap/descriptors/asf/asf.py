from ...logger import logger
from ...structure import Structure
from ..base import Descriptor
from .angular import AngularSymmetryFunction
from .radial import RadialSymmetryFunction
from typing import Union
import torch


class ASF(Descriptor):
  """
  Atomic Symmetry Function (ASF) descriptor.
  ASF is a vector of different radial and angular terms which describe the chemical environment of an atom.
  TODO: ASF should be independent of the input structure, but it should knows how to calculate the descriptor vector.
  See N2P2 -> https://compphysvienna.github.io/n2p2/topics/descriptors.html?highlight=symmetry%20function#
  """
  def __init__(self, element: str) -> None:
    self.element = element    # central element
    self._radial = []         # tuple(RadialSymmetryFunction , central_element, neighbor_element1)
    self._angular = []        # tuple(AngularSymmetryFunction, central_element, neighbor_element1, neighbor_element2)
    self.__cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6) # instantiate 

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
    elif isinstance(symmetry_function, AngularSymmetryFunction):
      self._angular.append((symmetry_function, self.element, neighbor_element1, neighbor_element2))
    else:
      msg = f"Unknown input symmetry function type"
      logger.error(msg)
      raise TypeError(msg)

  def __call__(self, structure:Structure, aid: int) -> torch.tensor: 
    """
    Calculate descriptor values for the input given structure.
    """
    # Update neighbor list first if needed
    if not structure.is_neighbor:
      structure.update_neighbor()

    x = structure.position            # tensor
    at = structure.atype              # tensor
    nn  = structure.neighbor.number   # tensor
    ni = structure.neighbor.index     # tensor
    emap= structure.element_map       # element map instance

    # Create output tensor
    result = torch.zeros(self.n_radial + self.n_angular, dtype=structure.dtype, device=structure.device)

    # Check aid atom type match the central element
    if not emap[self.element] == at[aid]:
      msg = f"Inconsistent central element ('{self.element}'): input aid={aid} ('{emap[int(at[aid])]}')"
      logger.error(msg)
      raise AssertionError(msg)

    # Get the list of neighboring atom indices
    ni_ = ni[aid, :nn[aid]]
    # Calculate the distances of neighboring atoms (detach flag must be disabled to keep the history of gradients)
    rij_ = structure.calculate_distance(aid, detach=False, neighbors=ni_)
    # Get the corresponding neighboring atom types
    tij_ = at[ni_] 
    
    # Loop over the radial terms
    for index, sf in enumerate(self._radial):
      # Find the neighboring atom indices that match the given ASF cutoff radius AND atom type
      ni_rc_ = (rij_ < sf[0].r_cutoff ).detach()
      ni_ = torch.nonzero( torch.logical_and(ni_rc_, tij_ == emap(sf[2]) ), as_tuple=True)[0]
      # Apply the ASF term kernels and sum over the neighboring atoms
      result[index] = torch.sum( sf[0].kernel(rij_[ni_] ), dim=0)

    # Loop over the angular terms
    for index, sf in enumerate(self._angular, start=self.n_radial):
      # Find neighboring atom indices that match the given ASF cutoff radius
      ni_rc_ = (rij_ < sf[0].r_cutoff ).detach()
      # Find atom indices of neighboring elements 1 and 2 
      ni_1_ = torch.nonzero( torch.logical_and(ni_rc_, tij_ == emap(sf[2])), as_tuple=True)[0]
      ni_2_ = torch.nonzero( torch.logical_and(ni_rc_, tij_ == emap(sf[3])), as_tuple=True)[0]
      # Apply the ASF term kernels and sum over the neighboring atoms
      rik = structure.calculate_distance(aid, neighbors=ni_2_)        # shape=(*)
      for j in ni_1_:
        Rij = x[aid] - x[j]                                           # shape=(3)
        Rik = x[aid] - x[ni_2_]                                       # shape=(*, 3)
        #Rjk = x[j]   - x[ni_2_]                                      # shape=(*, 3)
        # TODO: move cos calculation to structure
        cost =  self.__cosine_similarity(Rij.expand(Rik.shape), Rik)  # shape=(*)
        rij = structure.calculate_distance(aid, neighbors=j)          # shape=(1)
        rjk = structure.calculate_distance(j, neighbors=ni_2_)        # shape=(*)
        result[index] += torch.sum( sf[0].kernel(rij, rik, rjk, cost), dim=0) # broadcasting computation

    return result


  @property
  def n_radial(self) -> int:
    return len(self._radial)

  @property
  def n_angular(self) -> int:
    return len(self._angular)


