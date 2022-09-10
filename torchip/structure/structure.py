from ..logger import logger
from ..base import BaseTorchipClass
from ..config import dtype as _dtype
from ..config import device as _device
from ..utils.attribute import set_as_attribute
from ..utils.gradient import get_value
from .element import ElementMap
from .neighbor import Neighbor
from .box import Box
from typing import List, Dict, Tuple, Union
from torch import Tensor
import torch
# import structure_cpp


class Structure(BaseTorchipClass):
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
  _atomic_attributes = [
    'position',      # per-atom position x, y, and z
    'force',         # per-atom force components x, y, and z
    #'charge',       # per-atom electric charge
    'energy',        # per-atom energy
    'lattice',       # vectors of supercell 3x3 matrix
    'total_energy',  # total energy of atoms in simulation box
    #'total_charger' # total charge of atoms in simulation box
  ]
  _differentiable_atomic_attributes = [
    'position',      # force = -gradient(energy, position)
    #'charge, '      # TODO: for lang range interaction using charge models
  ]

  def __init__(self, 
      data: Dict = None,       
      r_cutoff: float = None,   
      dtype: torch.dtype = None,
      device: torch.device = None,
      requires_grad: bool = True,
      **kwargs,
    ) -> None: 
    """
    Initialize a structure including tensors, neighbor list atoms, simulation box, etc.

    :param data: A dictionary representation of atomic data including position, charge, energy, etc. 
    :type data: Dict
    :param dtype: Data type of internal tensors which represent structure, defaults to None
    :type dtype: torch.dtype, optional
    :param device: Device on which tensors are allocated, defaults to None
    :param r_cutoff: Cutoff radius for calculating the neighbor list, defaults to None
    :type r_cutoff: float, optional
    :type device: torch.device, optional
    :param position_grad: whether the position tensor is differentiable or not, defaults to True
    :type position_grad: bool, optional
    """
    self.dtype = dtype if dtype else _dtype.FLOAT
    self.device = device if device else _device.DEVICE
    self.requires_grad = requires_grad
    self.requires_neighbor_update = True
    # Additional input arguments
    self.element_map = kwargs.get("element_map", None)  # an instance of element map
    self.tensors = kwargs.get("tensors", None)          # tensors including position, energy, etc.
    self.box = kwargs.get("box", None)                  # an instance of simulation box
    self.neighbor = kwargs.get("neighbor", None)        # an instance of Neighbor atoms
    
    if data:
      try:
        self.element_map = ElementMap(data["element"]) 
        self._init_tensors(data)    
        self._init_box(data["lattice"])   
      except KeyError:
       logger.error(f"Cannot find one of the expected atomic attributes in input data:" \
                    f"{''.join(self._atomic_attributes)}", exception=KeyError)

    self._init_neighbor(r_cutoff)
    
    if self.tensors:
      set_as_attribute(self, self.tensors)

  def _init_neighbor(self, r_cutoff) -> None:
    """
    Initialize a neighbor list instance for a given cutoff radius. 
    
    It ignores updating the neighbor list if the new cutoff radius is the same the existing one.
    It's important to note that the Neighbor object in Structure is considered as a buffer and not 
    part of the atomic structure data. But, it uses for calculating descriptors, potential, etc. 
    It is the task of those classes to prepare the buffer neighbor for their own usage.  
    """
    if not self.neighbor:
      self.neighbor = Neighbor(r_cutoff)
      self.requires_neighbor_update = True 
      return

    if self.r_cutoff and self.r_cutoff == r_cutoff: 
      logger.debug(f"Skipping updating the neighbor list (cutoff radius): "
                   f"{self.r_cutoff} vs {r_cutoff} (new)")
      return

    self.neighbor.set_cutoff_radius(r_cutoff)
    self.requires_neighbor_update = True       

  def set_cutoff_radius(self, r_cutoff: float) -> None:
    """
    Set cutoff radius of the structure.
    This method is useful when having a potential with different cutoff radius.  

    :param r_cutoff: New cutoff radius
    :type r_cutoff: float
    """    
    self._init_neighbor(r_cutoff)

  def _init_box(self, lattice) -> None:
    """
    Create a simulation box object using provided lattice tensor.
    """
    if len(lattice) > 0:
      self.box = Box(lattice)  
    else:
      logger.debug("No lattice info were found in the structure")

  def _prepare_atype_tensor(self, elements: List) -> Tensor:
    """
    Set atom types using the element map
    """
    return torch.tensor([self.element_map(elem) for elem in elements], \
                          dtype=_dtype.INDEX, device=self.device)
    
  def _init_tensors(self, data: Dict) -> None:
    """
    Create tensors (allocate memory) from the input dictionary of structure data.
    It convert element (string) to atom type (integer) because of computational efficiency.
    """
    logger.debug("Allocating tensors for structure")
    self.tensors = {}
    try:
      # Tensors for atomic attributes
      for atomic_attr in self._atomic_attributes:
        if atomic_attr in self._differentiable_atomic_attributes:
          self.tensors[atomic_attr] = torch.tensor(data[atomic_attr], dtype=self.dtype, device=self.device, 
                                                   requires_grad=self.requires_grad)
        else:
          self.tensors[atomic_attr] = torch.tensor(data[atomic_attr], dtype=self.dtype, device=self.device)
      # A tensor for atomic type 
      self.tensors["atype"] = self._prepare_atype_tensor(data["element"])

    except KeyError:
      logger.error(f"Cannot find expected atomic attribute {atomic_attr} in the input data dictionary", exception=KeyError)

    # Logging
    for attr, tensor in self.tensors.items():
      logger.debug(f"{attr:12} -> Tensor(shape='{tensor.shape}', dtype='{tensor.dtype}', device='{tensor.device}')")
      
  def _tensors_to_data(self) -> Dict:
    """
    Cast the tensors to structure data.
    To be used for dumping structure into a file. 
    """
    data = {}
    for name, tensor in self.tensors.items():
      data[name] = get_value(tensor)
    return data

  def update_neighbor(self) -> None:
    """
    update neighbor list.
    This is a computationally expensive method.
    """
    self.neighbor.update(self)

  def reset_cutoff_radius(self, r_cutoff: float) -> None:
    """
    Reset cutoff radius of the neighbor list instance.

    :param r_cutoff: New cutoff radius
    :type r_cutoff: float
    """        
    self.neighbor.reset_cutoff_radius(r_cutoff)

  @staticmethod
  def _calculate_distance(
      pos: Tensor, 
      aid: int, 
      lattice: Tensor = None, 
      neighbors = None, 
      return_diff: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    [Kernel]
    Calculate a tensor of distances to all atoms existing in the structure from a specific atom. 
    TODO: input pbc flag, using default pbc from global configuration
    TODO: also see torch.cdist
    """   
    x = pos
    x = x[neighbors] if neighbors is not None else x 
    x = torch.unsqueeze(x, dim=0) if x.ndim == 1 else x  # when the neighbors index is a scalar
    dx = pos[aid] - x  

    # Apply PBC along x,y, and z directions if needed
    if lattice is not None:
        dx = Box._apply_pbc(dx, lattice) 

    # Calculate distance from dx tensor
    dis = torch.linalg.vector_norm(dx, dim=1)

    # Return results
    return dis if not return_diff else (dis, dx)

  def calculate_distance(self, aid: int, neighbors=None, return_diff=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Return a tensor of distances between a specific atom and all atoms existing in the structure. 
    """
    return Structure._calculate_distance(
        pos = self.position, 
        aid = aid, 
        lattice = self.box.lattice if self.box else None, # TODO: DRY
        neighbors = neighbors, return_diff=return_diff
      ) 

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
    return list({self.element_map(int(at)) for at in self.atype})

  @property
  def r_cutoff(self) -> float:
      return self.neighbor.r_cutoff

  def __repr__(self) -> str:
    return f"Structure(natoms={self.natoms}, elements={self.elements}, dtype={self.dtype}, device={self.device})"

  # def _copy_tensors(self) -> Dict[str, Tensor]:
  #   """
  #   Return a shallow copy of all tensors.
  #   :return: A dictionary of tensors for atomic data including position, energy, etc.
  #   :rtype: Dict[Tensors]
  #   """
  #   tensors_ = {}
  #   for attr, tensor in self.tensors.items():
  #     if attr in self._differentiable_atomic_attributes and self.requires_grad:
  #       tensors_[attr] = tensor #.detach().requires_grad_() # torch.clone
  #     else:
  #       tensors_[attr] = tensor #.detach() # torch.clone
  #   return tensors_

  # def copy(self, **kwargs):
  #   """
  #   Return a shallow copy of the structure (the tensors are not actually copied) with some adjusted settings
  #   possible from the input arguments. 
  #   """
  #   init_kwargs = {
  #     'r_cutoff'      : self.r_cutoff,
  #     'device'        : self.device,    
  #     'dtype'         : self.dtype,     
  #     'requires_grad' : self.requires_grad,              
  #     'element_map'   : self.element_map,
  #     'tensors'       : self._copy_tensors(),          
  #     'neighbor'      : self.neighbor,
  #     'box'           : self.box,
  #   }
  #   init_kwargs.update(kwargs)  # override the defaults
  #   structure_ = Structure(data=None, **init_kwargs)
  #   structure_.requires_neighbor_update = self.requires_neighbor_update
  #   return structure_