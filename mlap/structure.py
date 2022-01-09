
from.element import ElementMap
from typing import TextIO, Tuple, List
from collections import defaultdict
import torch


class Structure:
  """
  This class contains a collection of atoms in a box including position, forces, energy, cell, etc.   
  """

  def __init__(self):
    self._dict = None

  def _tokenize(self, line: str) -> Tuple[str, List[str]]:
    """
    Read the input line as a keyword and list of tokens.
    """
    tokens = line.rstrip("/n").split()
    if len(tokens) > 1:
      return (tokens[0].lower(), tokens[1:])
    elif len(tokens) > 0:
      return (tokens[0].lower(), None)
    else:
      return (None, None)

  def _cast_to_tensors(self) -> None:
    """
    Set structure dict info to the (pytorch) tensors.
    It should be defined along the read() method.
    """
    # TODO: logging
    dtype = torch.double
    device= torch.device("cpu")
    self.pos = torch.tensor(self._dict["position"], dtype=dtype, device=device, requires_grad=True)
    self.frc = torch.tensor(self._dict["force"], dtype=dtype, device=device)
    self.chg = torch.tensor(self._dict["charge"], dtype=dtype, device=device)
    self.eng = torch.tensor(self._dict["energy"], dtype=dtype, device=device)
    self.type = torch.tensor(self._dict["atom_type"], dtype=torch.int, device=device)
    self.cell = torch.tensor(self._dict["cell"], dtype=dtype, device=device)

  def read(self, file: TextIO) -> bool:
    """
    This method reads atomic configuration from the given input file.
    """
    # Read structure
    self._dict = defaultdict(list)
    while True:
      # Read one line from file
      line = file.readline()
      if not line:
        return False
      keyword, tokens = self._tokenize(line)
      # TODO: check begin keyword
      if keyword == "atom":
        self._dict["position"].append( [float(t) for t in tokens[:3]] )
        self._dict["element"].append( tokens[3] )
        self._dict["charge"].append( float(tokens[4]) )
        self._dict["energy"].append( float(tokens[5]) )
        self._dict["force"].append( [float(t) for t in tokens[6:9]] )
      elif keyword == "lattice":
        self._dict["cell"].append( [float(t) for t in tokens[:3]] )
      elif keyword == "energy":
        self._dict["total_energy"].append( float(tokens[0]) )
      elif keyword == "charge":
        self._dict["total_charge"].append( float(tokens[0]) )
      # TODO: what if it reaches EOF?
      elif keyword == "end": 
        break

    # Set atom types using element mapping
    element_map = ElementMap(self._dict["element"])
    self._dict["atom_type"] = [element_map[elem] for elem in self._dict["element"]] # TODO: optimize?

    # Create tensors
    #print(self._dict)
    self._cast_to_tensors()

    return True
      
  def write(self):
    pass

  def __str__(self):
    pass


