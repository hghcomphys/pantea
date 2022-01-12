
from .logger import logger
from .element import ElementMap
from .neighbor import Neighbor
from typing import List, Dict
from collections import defaultdict
import torch

dtype = torch.double
device = torch.device("cpu")


class Structure:
  """
  This class contains a collection of atoms in a box including position, forces, energy, cell, etc.   
  """

  def __init__(self, data: Dict[str, List]) -> None:
    """
    # Initializations
    """
    self._data = data
    self._tensors = defaultdict(None)
    self.element_map = None
    self.neighbor = Neighbor(r_cutoff=12.0) # TODO: cutoff value from descriptor
    
    # Tensors
    self._cast_to_tensors()
    self._set_tensors_as_attr()
    
    # Neighbor atoms
    self.neighbor.build(self)

  def _cast_to_tensors(self):
    """
    Cast structure data into the (pytorch) tensors.
    TODO: check the dictionary for missing items
    TODO: take care of some missing items.
    """
    # Direct casting
    self._tensors["position"] = torch.tensor(self._data["position"], dtype=dtype, device=device, requires_grad=True)
    self._tensors["force"] = torch.tensor(self._data["force"], dtype=dtype, device=device)
    self._tensors["charge"] = torch.tensor(self._data["charge"], dtype=dtype, device=device)
    self._tensors["energy"] = torch.tensor(self._data["energy"], dtype=dtype, device=device)
    self._tensors["cell"] = torch.tensor(self._data["cell"], dtype=dtype, device=device)

    # Set atom types using element mapping
    self.element_map = ElementMap(self._data["element"])
    atom_type = [self.element_map[elem] for elem in self._data["element"]] # TODO: optimize?
    self._tensors["atom_type"] = torch.tensor(atom_type, dtype=torch.long, device=device)

    for name, tensor in self._tensors.items():
      logger.info(
        f"Allocated '{name}' as a Tensor(shape='{tensor.shape}', dtype='{tensor.dtype}', device='{tensor.device}')")

  def _set_tensors_as_attr(self):
    logger.info(f"Setting {len(self._tensors)} tensors as attributes: {', '.join(self._tensors.keys())}")
    for name, tensor in self._tensors.items():
      setattr(self, name, tensor)
      
  def _cast_to_data(self) -> Dict[str, List]:
    """
    Cast the tensors to structure data. 
    """
    pass

  def __str__(self):
    pass
