from .logger import logger
from .config import CFG
from .element import ElementMap
from .neighbor import Neighbor
from .box import Box
from .utils.attribute import set_tensors_as_attr
from typing import List, Dict
from collections import defaultdict
import torch


class Structure:
  """
  This class contains a collection of atoms in a box including position, forces, energy, cell, etc.   
  Structure is unit atomic data which are used to calculate atomic descreptors.
  For the computational efficiency, vectors (more precisely tensors) of atomic data were considered 
  instead of each atom as a unit of data. 
  The most critical par of this class is calculating neighbor list and it can be done by inputing 
  an instance of Structure to the Neighbor class.
  For the MPI implementation, this class can be considerend as one domain in domain decomposition method.
  An C++ implementation might be required for MD simulation but not necessarily developing ML potential.     
  """
  def __init__(self, data: Dict[str, List], **param) -> None:
    """
    Initializations including tensors, neighbor atoms, and box.
    """ 
    self.device = param.get("device", CFG["device"])
    self.dtype = param.get("dtype", CFG["dtype"])

    self._tensors = defaultdict(None)
    self.element_map = None   # map element to atom type and vice versa.    

    # Neighboring atoms
    self.is_neighbor = False
    self.neighbor = Neighbor(r_cutoff=12.0) # TODO: cutoff value from descriptor
    # self.update_neighbor()     

    # Prepare tensors from input structure data
    self._cast_data_to_tensors(data)
    set_tensors_as_attr(self, self._tensors)
    
    # Create a box using the lattice matrix (useful for non-orthogonal lattice)
    self.box = Box(self.lattice) 
    
  def _cast_data_to_tensors(self, data: Dict[str, List]) -> None:
    """
    Cast a dictionary structure data into the (pytorch) tensors.
    It convert element (string) to atom type (integer) because of computational efficiency.
    TODO: check the input data dictionary for possibly missing items
    TODO: take care of some missing items.
    """
    # Direct casting
    self._tensors["position"] = torch.tensor(data["position"], dtype=self.dtype, device=self.device, requires_grad=True)
    self._tensors["force"] = torch.tensor(data["force"], dtype=self.dtype, device=self.device)
    self._tensors["charge"] = torch.tensor(data["charge"], dtype=self.dtype, device=self.device)
    self._tensors["energy"] = torch.tensor(data["energy"], dtype=self.dtype, device=self.device)
    self._tensors["lattice"] = torch.tensor(data["lattice"], dtype=self.dtype, device=self.device)

    # Set atom types using element mapping
    self.element_map = ElementMap(data["element"])
    atype = [self.element_map[elem] for elem in data["element"]] # TODO: optimize?
    self._tensors["atype"] = torch.tensor(atype, dtype=CFG["dtype_index"], device=self.device) # atom type

    # Logging existing tensors
    for name, tensor in self._tensors.items():
      logger.debug(
        f"Allocating '{name}' as a Tensor(shape='{tensor.shape}', dtype='{tensor.dtype}', device='{tensor.device}')")
      
  def _cast_tensors_to_data(self) -> Dict[str, List]:
    """
    Cast the tensors to structure data.
    To be used for dumping structure into a file. 
    """
    pass

  @property
  def natoms(self) -> int:
    return self._tensors["position"].shape[0]

  def update_neighbor(self) -> None:
    """
    update neighbor list.
    This is a computationally expensive method.
    """
    self.neighbor.update(self)

  def calculate_distance(self, aid: int, detach=False, neighbors=None) -> torch.Tensor:
    """
    This method calculates an array of distances of all atoms existing in the structure from an input atom. 
    TODO: input pbc flag, using default pbc from global configuration
    TODO: also see torch.cdist
    """
    def _apply_pbc(dx, l):   
      dx = torch.where(dx >  0.5*l, dx - l, dx)
      dx = torch.where(dx < -0.5*l, dx + l, dx)
      return dx

    x = self.position.detach() if detach else self.position
    x = x[neighbors] if neighbors is not None else x 
    x = torch.unsqueeze(x, dim=0) if x.ndim == 1 else x  # for when neighbors index is only a number
    dx = self.position[aid] - x

    # Apply PBC along x,y,and z directions
    dx[..., 0] = _apply_pbc(dx[..., 0], self.box.lx)
    dx[..., 1] = _apply_pbc(dx[..., 1], self.box.ly)
    dx[..., 2] = _apply_pbc(dx[..., 2], self.box.lz)

    # Calculate distance from dx tensor
    distance = torch.norm(dx, dim=1)

    return distance

  def select(self, element: str) -> torch.Tensor:
    """
    Return all atom ids with atom type same as the input element. 
    """
    return torch.nonzero( self.atype == self.element_map[element], as_tuple=True)[0].tolist()


