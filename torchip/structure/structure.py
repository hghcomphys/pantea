from ..logger import logger
from ..config import dtype, device
from ..utils.attribute import set_as_attribute
from ..utils.gradient import get_value
from .element import ElementMap
from .neighbor import Neighbor
from .box import Box
from typing import List, Dict
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
  __atomic_attributes = [
    'position',      # per-atom position x, y, and z
    'force',         # per-atom force components x, y, and z
    'charge',        # per-atom electric charge
    'energy',        # per-atom energy
    'lattice',       # vectors of supercell 3x3 matrix
    'total_energy',  # total energy of atoms in simulation box
    'total_charger'  # total charge of atoms in simulation box
  ]
  __differentiable_atomic_attributes = [
    'position',      # force = -gradient(energy, position)
    #'charge, '      # TODO: for lang range interaction using charge models
  ]

  def __init__(self, data: Dict = None, **kwargs) -> None: 
    """
    Initialization of tensors, neighbor atoms, and simulation box.

    :param data: A dictionary of atomic data including position, charge, energy, etc. 
    :type data: Dict
    """   
    # Set input arguments # TODO: loop over a class dict instead  
    self.device        = kwargs.get("device", device.DEVICE)    # data type float32, float64, etc.
    self.dtype         = kwargs.get("dtype", dtype.FLOATX)      # device either CPU or GPU
    self.requires_grad = kwargs.get("requires_grad", True)      # whether enabling autograd or not
    self.r_cutoff      = kwargs.get("r_cutoff", None)           # cutoff radius for calculating the neighbor list
    self.element_map   = kwargs.get("element_map", None)        # an instance of element map
    self.tensors       = kwargs.get("tensors", None)            # tensors including position, energy, etc.
    self.neighbor      = kwargs.get("neighbor", None)           # an instance of Neighbor atoms
    self.box           = kwargs.get("box", None)                # an instance of simulation box

    self.set_neighbor()
    if data is not None:
      try:
        self.element_map = ElementMap(data["element"]) 
        self.tensors = self._cast_data_to_tensors(data)    
        set_as_attribute(self, self.tensors)  
        self.set_box()    
      except KeyError:
       logger.error(f"Expected atomic attributes in input data: {''.join(self.__atomic_attributes)}")
        
  def set_neighbor(self):
    """
    Create a neighbor list for the given cutoff radius. 
    """
    if self.r_cutoff is not None:
      self.neighbor = Neighbor(self.r_cutoff)
      self.requires_neighbor_update = True 
    else:
      logger.debug("No cutoff radius was given, ignoring initializing the neighbor list of atoms") 

  def set_box(self):
    """
    Create a simulation box using provided lattice tensor.
    """
    if len(self.lattice) > 0:
      self.box = Box(self.lattice)  
    else:
      logger.debug("No lattice info were found in the structure")

  def _copy_tensors(self):
    """
    Return a shallow copy of all the tensors.

    :return: A dictionary of tensors
    :rtype: Dict[Tensors]
    """
    tensors_ = {}
    for name, tensor in self.tensors.items():
      if name in self.__differentiable_atomic_attributes and self.requires_grad:
        tensors_[name] = tensor #.detach().requires_grad_() # torch.clone
      else:
        tensors_[name] = tensor #.detach() # torch.clone

    return tensors_

  def copy(self, **kwargs):
    """
    Return a shallow copy of the structure (the tensors are not actually copied) with some adjusted settings
    possible from the input arguments. 
    """
    init_kwargs = {
      'device'        : self.device,    
      'dtype'         : self.dtype,     
      'requires_grad' : self.requires_grad,
      'r_cutoff'      : self.r_cutoff,                   
      'element_map'   : self.element_map,
      'tensors'       : self._copy_tensors(),          
      'neighbor'      : self.neighbor,
      'box'           : self.box,
    }
    init_kwargs.update(kwargs)  # override the defaults

    structure_ = Structure(data=None, **init_kwargs)
    set_as_attribute(structure_, structure_.tensors)

    return structure_
    
  def set_r_cutoff(self, r_cutoff: float) -> None:
    """
    Set cutoff radius and then update the neighbor list accordingly.

    Args:
        r_cutoff (float): New cutoff radius
    """
    if r_cutoff is None:
      self.r_cutoff = None
      self.neighbor = None
      return

    if (self.r_cutoff is None) or (self.r_cutoff < r_cutoff):
      self.r_cutoff = r_cutoff
      self.neighbor = Neighbor(self.r_cutoff)
      self.requires_neighbor_update = True
      logger.debug(f"Resetting the cutoff radius of structure: new r_cutoff={self.r_cutoff}")

  def _prepare_atype_tensor(self, elements: List) -> Tensor:
    """
    Set atom types using the element map
    """
    atype = [self.element_map(elem) for elem in elements]
    return torch.tensor(atype, dtype=dtype.INDEX, device=self.device)
    
  def _cast_data_to_tensors(self, data: Dict) -> None:
    """
    Cast a dictionary structure data into the (pytorch) tensors.
    It convert element (string) to atom type (integer) because of computational efficiency.
    TODO: check the input data dictionary for possibly missing items
    """
    tensors_ = {}

    try:
      # Tensors for atomic attributes
      for atomic_attr in self.__atomic_attributes:
        if atomic_attr in self.__differentiable_atomic_attributes:
          tensors_[atomic_attr] = torch.tensor(data[atomic_attr], dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
        else:
          tensors_[atomic_attr] = torch.tensor(data[atomic_attr], dtype=self.dtype, device=self.device)
      # A tensor for atomic type 
      tensors_["atype"] = self._prepare_atype_tensor(data["element"])
      
    except KeyError:
      logger.error(f"Cannot find expected atomic attribute {atomic_attr} in the input data dictionary", exception=KeyError)

    # Logging
    for name, tensor in tensors_.items():
      logger.debug(
        f"Allocated '{name}' as a Tensor(shape='{tensor.shape}', dtype='{tensor.dtype}', device='{tensor.device}')"
      )

    return tensors_
      
  def _cast_tensors_to_data(self) -> Dict:
    """
    Cast the tensors to structure data.
    To be used for dumping structure into a file. 
    """
    dict_ = {}
    for name, tensor in self.tensors.items():
      dict_[name] = get_value(tensor)

    return dict_

  def update_neighbor(self) -> None:
    """
    update neighbor list.
    This is a computationally expensive method.
    """
    if self.neighbor:
      self.neighbor.update(self)
    else:
      logger.error(f"No cutoff radius for structure is given yet: r_cutoff={self.r_cutoff}", exception=ValueError)

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
      dx = self.apply_pbc(dx) # using broadcasting

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
    return self.tensors["position"].shape[0]

  @property
  def elements(self) -> List[str]:
    return list({self.element_map[int(at)] for at in self.atype})

  def __repr__(self) -> str:
    return f"Structure(natoms={self.natoms}, elements={self.elements}, dtype={self.dtype}, device={self.device})"