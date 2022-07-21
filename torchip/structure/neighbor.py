from ..logger import logger
from ..config import dtype, device
from ..utils.attribute import set_as_attribute
from collections import defaultdict
# from .structure import Structure  # TODO: circular import error
import torch


class Neighbor:
  """
  Neighbor class creates a neighbor list of atom for the input structure.
  It is designed to be independent of the input structure. 
  For MD simulations, re-neighboring the list is required every few steps (e.g. by defining a skin radius). 
  """
  def __init__(self, r_cutoff: float): 
    self.r_cutoff = r_cutoff
    self.tensors = {}
    logger.debug(f"{self.__class__.__name__}(r_cutoff={self.r_cutoff})")

  def update(self, structure) -> None:
    """
    This method updates the neighbor atom tensors including the number of neighbor and neighbor atom indices 
    within the input structure.
    
    TODO: optimize updating the neighbor list, for example using the cell mesh, bin atoms (miniMD), etc.
    TODO: reduce natoms*natoms tensor size!
    TODO: define max_num_neighbor to avoid extra memory allocation!
    """    
    if not structure.requires_neighbor_update:
      logger.debug("Skip updating the neighbor list")
      return

    # Neighbor atoms numbers and indices
    self.tensors["number"] = torch.empty(structure.natoms, dtype=dtype.UINT, device=structure.device)
    self.tensors["index"] = torch.empty(structure.natoms, structure.natoms, dtype=dtype.INDEX, device=structure.device) 
    set_as_attribute(self, self.tensors)

    nn = self.number
    ni = self.index
    for aid in range(structure.natoms): # TODO: optimization: torch unbind or vmap
      rij = structure.calculate_distance(aid, detach=False)
      # Get atom indices within the cutoff radius
      ni_ = torch.nonzero( rij < self.r_cutoff, as_tuple=True)[0]
      # Remove self-counting atom index
      ni_ = ni_[ni_ != aid] 
      # Set neighbor list tensors
      nn[aid] = ni_.shape[0]
      ni[aid, :nn[aid]] = ni_ 

    # Avoid updating the neighbor list the next time
    structure.requires_neighbor_update = False


