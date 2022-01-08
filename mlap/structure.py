
from typing import TextIO, Tuple, List
from collections import defaultdict


class Structure:
  """
  This class contains a collection of atoms in a box including position, forces, energy, cell, etc.   
  """

  def __init__(self):
    pass

  # @staticmethod
  # def prefetch(file: TextIO):
  #   n_atoms = 0
  #   is_begin = False
  #   ref_pos = file.tell()
  #   line = file.readline()
  #   while line:
  #     keyword = line.rstrip("/n").split()[0].lower()
  #     if keyword == "begin":
  #       is_begin = True
  #     elif is_begin and ( keyword == "atom"):
  #       n_atoms += 1
  #     elif keyword == "end":
  #       is_begin = False
  #       break
  #   file.seek(ref_pos)
  #   return n_atoms

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

  def read(self, file: TextIO) -> bool:
    """
    This method reads atomic configuration from the given input file.
    """
    dict_ = defaultdict(list)
    while True:
      # Read one line from file
      line = file.readline()
      if not line:
        return False
      keyword, tokens = self._tokenize(line)
      # TODO: check begin keyword
      if keyword == "atom":
        dict_["position"].append( [float(t) for t in tokens[:3]] )
        dict_["element"].append( tokens[3] )
        dict_["charge"].append( float(tokens[4]) )
        dict_["energy"].append( float(tokens[5]) )
        dict_["position"].append( [float(t) for t in tokens[6:9]] )
      elif keyword == "lattice":
        dict_["cell"].append( [float(t) for t in tokens[:3]] )
      elif keyword == "energy":
        dict_["total_energy"].append( float(tokens[0]) )
      elif keyword == "charge":
        dict_["total_charge"].append( float(tokens[0]) )
      # TODO: what if it reaches EOF?
      elif keyword == "end": 
        break

    # TODO: convert to pytorch tensors
    print(dict_)
    return True
      
  def write(self):
    pass

  def __str__(self):
    pass


