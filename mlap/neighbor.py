from .box import Box
import torch


device = torch.device("cpu")

class Neighbor:
  """
  Calculate list of neighbor atoms for the given structure.
  """
  def __init__(self, r_cutoff: float): 
    # TODO: cutoff
    self.r_cutoff = r_cutoff
    self.nn = None    # number of neighbors for each atom
    self.ngb = None   # neighbor indices for atoms
    pass

  def build(self, structure):
    """
    Build neighbor atoms
    """
    n_atoms = structure.pos.shape[0]
    box = Box(structure.cell)
    self.nn = torch.zeros(n_atoms, dtype=torch.int, device=device)
    self.ngb = torch.zeros(n_atoms, n_atoms, dtype=torch.int, device=device)

    # TODO: optimization
    x = structure.pos.detach()
    for aid in range(n_atoms):
      distances_ = torch.norm(x-x[aid], dim=1)
      neighbors_ = torch.nonzero( distances_ < self.r_cutoff, as_tuple=True)[0].tolist()
      neighbors_.remove(aid)  # remove self-neighboring
      self.nn[aid] = len(neighbors_)
      for i, neighbor_index in enumerate(neighbors_):
        self.ngb[aid, i] = neighbor_index 

    # print(self.nn)
    # print(self.ngb)






