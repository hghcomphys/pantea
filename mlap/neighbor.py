from .logger import logger
from .box import Box
from collections import defaultdict
import torch

device = torch.device("cpu")


class Neighbor:
  """
  Calculate a list of neighbor atoms for a structure.
  """
  def __init__(self, r_cutoff: float): 
    # TODO: cutoff
    self.r_cutoff = r_cutoff
    self._tensors = defaultdict()
    self.nn = None    # number of neighbors for each atom
    self.ngb = None   # neighbor indices for atoms
    pass

  def build(self, structure):
    """
    Build neighbor atoms
    """
    natoms = structure.position.shape[0]
    box = Box(structure.cell)
    self._tensors["numbers"] = torch.empty(natoms, dtype=torch.long, device=device)
    self._tensors["indices"] = torch.empty(natoms, natoms, dtype=torch.long, device=device) # TODO: natoms*natoms?
    self._set_tensors_as_attr()

    # TODO: optimization
    x = structure.position.detach()
    for aid in range(natoms):
      distances_ = torch.norm(x-x[aid], dim=1)
      neighbors_ = torch.nonzero( distances_ < self.r_cutoff, as_tuple=True)[0].tolist()
      neighbors_.remove(aid)  # remove self-counting
      self.numbers[aid] = len(neighbors_)
      for i, neighbor_index in enumerate(neighbors_):
        self.indices[aid, i] = neighbor_index 

  def _set_tensors_as_attr(self):
    logger.info(f"Setting {len(self._tensors)} tensors as attributes: {', '.join(self._tensors.keys())}")
    for name, tensor in self._tensors.items():
      setattr(self, name, tensor)






