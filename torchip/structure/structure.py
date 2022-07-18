from ..logger import logger
from ..config import dtype, device
from .element import ElementMap
from .neighbor import Neighbor
from .box import Box
from ..utils.attribute import set_tensors_as_attr
from typing import List, Dict
from collections import defaultdict
from torch import Tensor
import torch
# import structure_cpp


class Structure:
  """
  Structure class contains arrays of atomic information including position, forces, energy, cell, and  more) 
  for a collection of atoms in a simulation box.   
  An instance of the Structure class is an unit of atomic data which being used to calculate the (atomic) descreptors.
  For computational reasons, vectors, more precisely tensors, of atomic data are used instead of defining 
  individual atoms as a unit of the atomic data. 

  The most computationally expensive section of this class is when calculating the neighbor list. 
  This task is done by giving an instance of Structure to the Neighbor class which is responsible 
  for updating the neighbor lists.

  For the MPI implementation, this class can be considered as one domain in domain decomposition method (see miniMD code).
  An C++ implementation might be required for MD simulation but not necessarily developing ML potential.   

  Also tensor for atomic positions (and probably atomic change in future) has to be differentiable and this requires
  keeping track of all operations in the computational graph that can lead ot to large memory usage. 
  Some methods are intoduced here to avoid gradient whenever it's possible.  
  """
  # TODO: mesh gird method can be used to seed up of creating/updating the neighbot list.

  def __init__(self, data: Dict, **kwargs) -> None:
    """
    Initialization of tensors, neighbor atoms, and simulation box.
    """ 
    # Set dtype and device
    self.device = kwargs.get("device", device.DEVICE)
    self.dtype = kwargs.get("dtype", dtype.FLOATX)

    # Whether keeping the history of gradients (e.g. position tensor) or not
    self.requires_grad = kwargs.get("requires_grad", True)

    self._tensors = defaultdict(None)   # an default dictionary of torch tensors
    self.element_map = None             # map an element to corrresponsing atom type and vice versa.    

    # Neighboring atoms
    self.r_cutoff = kwargs.get("r_cutoff", None) 
    self.neighbor = Neighbor(self.r_cutoff) if self.r_cutoff else None
    self.is_neighbor = False   

    # Prepare tensors from input structure data
    self._cast_data_to_tensors(data)
    set_tensors_as_attr(self, self._tensors)
    
    # Create a box using the lattice matrix (useful for non-orthogonal lattice)
    if len(self.lattice) != 0:
      self.box = Box(self.lattice)  
    else:
      self.box = None 
      logger.debug("No lattice info were found in structure")

  def set_r_cutoff(self, r_cutoff: float) -> None:
    """
    Set cutoff radius and then update the neighbor list accordingly.

    Args:
        r_cutoff (float): New cutoff radius
    """
    if r_cutoff is None:
      self.r_cutoff = None
      self.neighbor = None
      self.is_neighbor = False
      return

    if (self.r_cutoff is None) or (self.r_cutoff < r_cutoff):
      self.r_cutoff = r_cutoff
      self.neighbor = Neighbor(self.r_cutoff)
      self.is_neighbor = False
      logger.debug(f"Resetting cutoff radius of structure: r_cutoff={self.r_cutoff}")
    
  def _cast_data_to_tensors(self, data: Dict) -> None:
    """
    Cast a dictionary structure data into the (pytorch) tensors.
    It convert element (string) to atom type (integer) because of computational efficiency.
    TODO: check the input data dictionary for possibly missing items
    TODO: take care of some missing items.
    """
    # Direct casting
    self._tensors["position"] = torch.tensor(data["position"], dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
    self._tensors["force"] = torch.tensor(data["force"], dtype=self.dtype, device=self.device)
    self._tensors["charge"] = torch.tensor(data["charge"], dtype=self.dtype, device=self.device) # TODO: add requires_grad
    self._tensors["energy"] = torch.tensor(data["energy"], dtype=self.dtype, device=self.device)
    self._tensors["lattice"] = torch.tensor(data["lattice"], dtype=self.dtype, device=self.device)
    self._tensors["total_energy"] = torch.tensor(data["total_energy"], dtype=self.dtype, device=self.device)
    self._tensors["total_charge"] = torch.tensor(data["total_charge"], dtype=self.dtype, device=self.device)

    # Set atom types using element mapping
    self.element_map = ElementMap(data["element"])
    atype = [self.element_map[elem] for elem in data["element"]] # TODO: optimize?
    self._tensors["atype"] = torch.tensor(atype, dtype=dtype.INDEX, device=self.device) # atom type

    # Logging existing tensors
    for name, tensor in self._tensors.items():
      logger.debug(
        f"Allocating '{name}' as a Tensor(shape='{tensor.shape}', dtype='{tensor.dtype}', device='{tensor.device}')"
      )
      
  def _cast_tensors_to_data(self) -> Dict:
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
    if self.neighbor:
      self.neighbor.update(self)
    else:
      logger.error("No cutoff radius is given", exception=ValueError)

  @staticmethod
  def _apply_pbc(dx: Tensor, l: float) -> Tensor:
    """
    An utility and static method to apply PBC along a specific direction. 
    """   
    dx = torch.where(dx >  0.5E0*l, dx - l, dx)
    dx = torch.where(dx < -0.5E0*l, dx + l, dx)
    return dx
    # return structure_cpp._apply_pbc(dx, l)

  def apply_pbc(self, dx: Tensor) -> Tensor: 
    """
    This method applies PBC on the input array (assuming position difference).
    """
    # Apply PBC along x,y,and z directions
    dx[..., 0] = self._apply_pbc(dx[..., 0], self.box.lx) # x
    dx[..., 1] = self._apply_pbc(dx[..., 1], self.box.ly) # y
    dx[..., 2] = self._apply_pbc(dx[..., 2], self.box.lz) # z
    # dx[..., 0] = structure_cpp._apply_pbc(dx[..., 0], self.box.lx) # x
    # dx[..., 1] = structure_cpp._apply_pbc(dx[..., 1], self.box.ly) # y
    # dx[..., 2] = structure_cpp._apply_pbc(dx[..., 2], self.box.lz) # z
    return dx
    # TODO: non-orthogonal box
    # return structure_cpp.apply_pbc(dx, torch.diagonal(self.box.lattice)) 

  def calculate_distance(self, aid: int, detach=False, neighbors=None, difference=False) -> Tensor: # TODO: also tuple?
    """
    This method calculates an array of distances of all atoms existing in the structure from an input atom. 
    TODO: input pbc flag, using default pbc from global configuration
    TODO: also see torch.cdist
    """
    x = self.position.detach() if detach else self.position
    x = x[neighbors] if neighbors is not None else x 
    x = torch.unsqueeze(x, dim=0) if x.ndim == 1 else x  # for when neighbors index is only a number
    dx = self.position[aid] - x

    # Apply PBC along x,y,and z directions if lattice info is provided 
    if self.box is not None:
      # dx[..., 0] = self._apply_pbc(dx[..., 0], self.box.lx)
      # dx[..., 1] = self._apply_pbc(dx[..., 1], self.box.ly)
      # dx[..., 2] = self._apply_pbc(dx[..., 2], self.box.lz)
      dx = self.apply_pbc(dx) # broadcasting instead

    # Calculate distance from dx tensor
    distance = torch.linalg.vector_norm(dx, dim=1)

    return distance if not difference else (distance, dx)

  def select(self, element: str) -> Tensor:
    """
    Return all atom ids with atom type same as the input element. 
    """
    return torch.nonzero(self.atype == self.element_map[element], as_tuple=True)[0]

  @property
  def natoms(self) -> int:
    return self.position.shape[0]

  @property
  def elements(self) -> List[str]:
    return list({self.element_map[int(at)] for at in self.atype})

  def __str__(self) -> str:
    return f"Structure: natoms={self.natoms}, elements={self.elements}"