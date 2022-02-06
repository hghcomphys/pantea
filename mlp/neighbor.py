from .logger import logger
from .config import CFG
from .utils.attribute import set_tensors_as_attr
from collections import defaultdict
# from .structure import Structure # TODO: circular import error
import torch


class Neighbor:
  """
  This class creates neighbor list of atom for an input structure.
  It basically should be independent of the input structure. 
  For MD simulations, re-neighboring the list is required every few steps. 
  TODO: optimize way to update the list, for example using the cell mesh, bin atoms (miniMD), etc.
  TODO: natoms*natoms tensor size?!
  """
  def __init__(self, r_cutoff: float): 
    self.r_cutoff = r_cutoff
    self._tensors = defaultdict(None)

  def update(self, structure) -> None:
    """
    This method updates the neighbor atom tensors such as number of neighbor and neighbor atom indices 
    inside the input structure.
    TODO: define max_num_neighbor to avoid extra memory allocation!
    """    
    if not structure.is_neighbor:
      structure.is_neighbor = True

      # Neighbor atoms numbers and indices
      # TODO: torch._resize, use idea of max_num_neighbor
      self._tensors["number"] = torch.empty(structure.natoms, dtype=CFG["dtype_index"], device=structure.device)
      self._tensors["index"] = torch.empty(structure.natoms, structure.natoms, dtype=CFG["dtype_index"], device=structure.device) 
      set_tensors_as_attr(self, self._tensors)

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
    else:
      logger.warning("Skiping to update neighboring atoms in the structure")





