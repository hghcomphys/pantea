
from.element import ElementMap
from.neighbor import Neighbor
from typing import TextIO, Tuple, List, Dict
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
    self._element_map = None
    self._neighbor = Neighbor(r_cutoff=12.0)

    self._cast_to_tensors()
    # self._neighbor.build(self)

  def _cast_to_tensors(self):
    """
    Cast structure data into the (pytorch) tensors.
    TODO: check the dictionary for missing items
    TODO: take care of some missing items.
    TODO: logging
    """
    # Direct casting
    self._tensors["position"] = torch.tensor(self._data["position"], dtype=dtype, device=device, requires_grad=True)
    self._tensors["force"] = torch.tensor(self._data["force"], dtype=dtype, device=device)
    self._tensors["charge"] = torch.tensor(self._data["charge"], dtype=dtype, device=device)
    self._tensors["energy"] = torch.tensor(self._data["energy"], dtype=dtype, device=device)
    self._tensors["cell"] = torch.tensor(self._data["cell"], dtype=dtype, device=device)

    # Set atom types using element mapping
    self._element_map = ElementMap(self._data["element"])
    atom_type = [self._element_map[elem] for elem in self._data["element"]] # TODO: optimize?
    self._tensors["atom_type"] = torch.tensor(atom_type, dtype=torch.int, device=device)
      
  def _cast_to_data(self):
    """
    Cast the tensors to structure data (dictionary). 
    """
    pass

  def __str__(self):
    pass
