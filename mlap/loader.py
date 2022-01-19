from typing import Tuple, List, TextIO, Dict
from collections import defaultdict
from pathlib import Path


class StructureLoader:
  """
  A base class which loads atom structure in different (file) formats. 
  For now it's only implemented for RuNNer file format.
  TODO: logging
  """

  def __init__(self, filename: Path) -> None:
    self.filename = Path(filename)
    self._data = None

  def get_data(self) -> Dict[str, List]:
    """
    A generator method which returns a each data structure as dictionary.
    """
    with open(str(self.filename), "r") as file:
      if self.read(file):
        yield self._data

  def _tokenize(self, line: str) -> Tuple[str, List[str]]:
    """
    Read the input line as a keyword and list of tokens.
    An utility method. 
    """
    tokens = line.rstrip("/n").split()
    if len(tokens) > 1:
      return (tokens[0].lower(), tokens[1:])
    elif len(tokens) > 0:
      return (tokens[0].lower(), None)
    else:
      return (None, None)

  def read(self, file: TextIO) -> bool:
    """
    This method reads the next structure from the given input file.
    """
    self._data = defaultdict(list)
    # Read next structure
    while True:
      # Read one line from file
      line = file.readline()
      if not line:
        return False
      keyword, tokens = self._tokenize(line)
      # TODO: check begin keyword
      if keyword == "atom":
        self._data["position"].append( [float(t) for t in tokens[:3]] )
        self._data["element"].append( tokens[3] )
        self._data["charge"].append( float(tokens[4]) )
        self._data["energy"].append( float(tokens[5]) )
        self._data["force"].append( [float(t) for t in tokens[6:9]] )
      elif keyword == "lattice":
        self._data["lattice"].append( [float(t) for t in tokens[:3]] )
      elif keyword == "energy":
        self._data["total_energy"].append( float(tokens[0]) )
      elif keyword == "charge":
        self._data["total_charge"].append( float(tokens[0]) )
      # TODO: what if it reaches EOF?
      elif keyword == "end": 
        break
    return True