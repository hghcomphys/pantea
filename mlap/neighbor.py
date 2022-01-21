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
    """    
    if not structure.is_neighbor:
      structure.is_neighbor = True
      x = structure.position.detach()
      nn = structure.neighbor_number
      ngb = structure.neighbor_index
      for aid in range(structure.natoms): # TODO: optimization: torch unbind or vmap
        distances_ = torch.norm(x-x[aid], dim=1) # TODO: apply PBC
        neighbors_ = torch.nonzero( distances_ < self.r_cutoff, as_tuple=True)[0].tolist()
        neighbors_.remove(aid)  # remove self-counting
        nn[aid] = len(neighbors_)
        for i, neighbor_index in enumerate(neighbors_):
          ngb[aid, i] = neighbor_index 
    else:
      logger.warning("Skiping to update neighboring atoms in the structure")





