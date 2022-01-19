from .logger import logger
# from .structure import Structure # TODO: circular import error
import torch

device = torch.device("cpu")


class Neighbor:
  """
  Calculate a list of neighbor atoms for a structure.
  TODO: Optimization update() method using cell mesh method, bining, etc.
  """
  def __init__(self, r_cutoff: float): 
    # TODO: cutoff
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
      for aid in range(structure.natoms):
        distances_ = torch.norm(x-x[aid], dim=1) # TODO: apply PBC
        neighbors_ = torch.nonzero( distances_ < self.r_cutoff, as_tuple=True)[0].tolist()
        neighbors_.remove(aid)  # remove self-counting
        nn[aid] = len(neighbors_)
        for i, neighbor_index in enumerate(neighbors_):
          ngb[aid, i] = neighbor_index 
    else:
      logger.warning("Skiping to update neighboring atoms in the structure")





