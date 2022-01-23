from .logger import logger
# from .structure import Structure # TODO: circular import error
import torch


class Neighbor:
  """
  This class creates neighbor list of atom for an input structure.
  It basically should be independent of the input structure. 
  For MD simulations, re-neighboring the list is required every few steps. 
  TODO: optimize way to update the list, for example using the cell mesh, bin atoms (miniMD), etc.
  TODO: is the any benefit to move neighbor list tensors from structure to here?
  """
  def __init__(self, r_cutoff: float): 
    self.r_cutoff = r_cutoff

  def update(self, structure) -> None:
    """
    This method updates the neighbor atom tensors such as number of neighbor and neighbor atom indices 
    inside the input structure.
    TODO: define max_num_neighbor to avoid extra memory allocation!
    """    
    if not structure.is_neighbor:
      structure.is_neighbor = True
      nn = structure.neighbor_number
      ni = structure.neighbor_index
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





