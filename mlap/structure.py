
from .logger import logger
from .element import ElementMap
from .neighbor import Neighbor
from .box import Box
from typing import List, Dict
from collections import defaultdict
import torch

# TODO: define a global config for device and dtype
dtype = torch.double
device = torch.device("cpu")


class Structure:
  """
  This class contains a collection of atoms in a box including position, forces, energy, cell, etc.   
  """
  def __init__(self, data: Dict[str, List]) -> None:
    """
    Initializations including tensors, neighbor atoms, and box.
    """
    self._data = data
    self._tensors = defaultdict(None)
    self.neighbor = Neighbor(r_cutoff=12.0) # TODO: cutoff value from descriptor
    self.element_map = None   # map element to atom type and vice versa.
    self.box = None           # box info, useful for non-orthogonal lattice
    self.is_neighbor = False
    
    # Tensors
    self._cast_data_to_tensors()
    self._set_tensors_as_attr()
    
    # Create a box using the lattice matrix 
    self.box = Box(self.lattice) 
    
    # Find neighboring atoms
    self.update_neighbor()

  def _cast_data_to_tensors(self):
    """
    Cast structure data into the (pytorch) tensors.
    TODO: check the input data dictionary for possibly missing items
    TODO: take care of some missing items.
    """
    # Direct casting
    self._tensors["position"] = torch.tensor(self._data["position"], dtype=dtype, device=device, requires_grad=True)
    self._tensors["force"] = torch.tensor(self._data["force"], dtype=dtype, device=device)
    self._tensors["charge"] = torch.tensor(self._data["charge"], dtype=dtype, device=device)
    self._tensors["energy"] = torch.tensor(self._data["energy"], dtype=dtype, device=device)
    self._tensors["lattice"] = torch.tensor(self._data["lattice"], dtype=dtype, device=device)

    # Set atom types using element mapping
    self.element_map = ElementMap(self._data["element"])
    atom_type = [self.element_map[elem] for elem in self._data["element"]] # TODO: optimize?
    self._tensors["atom_type"] = torch.tensor(atom_type, dtype=torch.long, device=device)

    # Neighbor atoms info
    self._tensors["neighbor_number"] = torch.empty(self.natoms, dtype=torch.long, device=device)
    self._tensors["neighbor_index"] = torch.empty(self.natoms, self.natoms, dtype=torch.long, device=device) # TODO: natoms*natoms?

    # Logging existing tensors
    for name, tensor in self._tensors.items():
      logger.info(
        f"Allocated '{name}' as a Tensor(shape='{tensor.shape}', dtype='{tensor.dtype}', device='{tensor.device}')")

  def _set_tensors_as_attr(self):
    logger.info(f"Setting {len(self._tensors)} tensors as '{self.__class__.__name__}'"
                f" class attributes: {', '.join(self._tensors.keys())}")
    for name, tensor in self._tensors.items():
      setattr(self, name, tensor)
      
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
    self.neighbor.update(self)
