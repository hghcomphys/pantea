from ..logger import logger
from ..config import CFG
from ..element import ElementMap
from ..neighbor import Neighbor
from ..box import Box
from ..utils.attribute import set_tensors_as_attr
from typing import List, Dict
from collections import defaultdict
import torch
import structure_cpp


class Structure:
  """
  This class contains arrays of atomic information (i.e. position, forces, energy, cell, and  more) for a collection of atoms.   
  An instance of the Structure class is an unit of atomic data which being used to calculate the (atomic) descreptors.
  For computational reasons, vectors (more precisely tensors) of atomic data are used instead of defining 
  individual atoms as a unit of our atomic data. 

  The most computationally expensive section of this class is when calculating the neighbor list. 
  This task is done by giving an instance of Structure to the Neighbor class which is responsible for updating the neighbor lists.
  TODO: mesh gird methid can be used to seed up of creating/updating the neighbot list.

  For the MPI implementation, this class can be considerend as one domain in domain decomposition method (see miniMD code).
  An C++ implementation might be required for MD simulation but not necessarily developing ML potential.   
  """
  _default_r_cutoff = 12.0 # TODO: move to CFG?

  def __init__(self, data: Dict[str, List], **kwargs) -> None:
    """
    Initialization of tensors, neighbor atoms, and box.
    """ 
    # Set dtype and device
    self.device = kwargs.get("device", CFG["device"])
    self.dtype = kwargs.get("dtype", CFG["dtype"])

    # Whether keep the history of gradients (e.g. position tensor) due to computational efficiency
    self.requires_grad = kwargs.get("requires_grad", True)

    self._tensors = defaultdict(None)   # an default dictionary of torch tensors
    self.element_map = None             # map an element to corrresponsing atom type and vice versa.    

    # Neighboring atoms
    self.is_neighbor = False
    self.r_cutoff = kwargs.get("r_cutoff", self._default_r_cutoff) 

    self.neighbor = Neighbor(r_cutoff=self.r_cutoff) 
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
    self._tensors["position"] = torch.tensor(data["position"], dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
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

  @staticmethod
  def _apply_pbc(dx: torch.Tensor, l: float) -> torch.Tensor:
    """
    An utility and static method to apply PBC along a specific direction. 
    """   
    # dx = torch.where(dx >  0.5E0*l, dx - l, dx)
    # dx = torch.where(dx < -0.5E0*l, dx + l, dx)
    # return dx
    return structure_cpp._apply_pbc(dx, l)

  def apply_pbc(self, dx: torch.Tensor) -> torch.Tensor: 
    """
    This method applies PBC on the input array (assuming position difference).
    """
    # Apply PBC along x,y,and z directions
    # dx[..., 0] = self._apply_pbc(dx[..., 0], self.box.lx) # x
    # dx[..., 1] = self._apply_pbc(dx[..., 1], self.box.ly) # y
    # dx[..., 2] = self._apply_pbc(dx[..., 2], self.box.lz) # z
    # dx[..., 0] = structure_cpp._apply_pbc(dx[..., 0], self.box.lx) # x
    # dx[..., 1] = structure_cpp._apply_pbc(dx[..., 1], self.box.ly) # y
    # dx[..., 2] = structure_cpp._apply_pbc(dx[..., 2], self.box.lz) # z
    # return dx
    # TODO: does _apply_pbc really works because of broadcasting?
    return structure_cpp._apply_pbc(dx, torch.diagonal(self.box.lattice)) 

  def calculate_distance(self, aid: int, detach=False, neighbors=None, difference=False) -> torch.Tensor: # TODO: also tuple?
    """
    This method calculates an array of distances of all atoms existing in the structure from an input atom. 
    TODO: input pbc flag, using default pbc from global configuration
    TODO: also see torch.cdist
    """
    x = self.position.detach() if detach else self.position
    x = x[neighbors] if neighbors is not None else x 
    x = torch.unsqueeze(x, dim=0) if x.ndim == 1 else x  # for when neighbors index is only a number
    dx = self.position[aid] - x

    # Apply PBC along x,y,and z directions  # TODO: replacing by self.apply_pbc
    # dx[..., 0] = self._apply_pbc(dx[..., 0], self.box.lx)
    # dx[..., 1] = self._apply_pbc(dx[..., 1], self.box.ly)
    # dx[..., 2] = self._apply_pbc(dx[..., 2], self.box.lz)
    dx = self.apply_pbc(dx)

    # Calculate distance from dx tensor
    distance = torch.linalg.vector_norm(dx, dim=1)

    return distance if not difference else (distance, dx)

  def select(self, element: str) -> torch.Tensor:
    """
    Return all atom ids with atom type same as the input element. 
    """
    return torch.nonzero(self.atype == self.element_map[element], as_tuple=True)[0]


