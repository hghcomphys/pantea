from ...logger import logger
from ...utils.tokenize import tokenize
from ...structure.structure import Structure
from ..base import StructureDataset
from typing import Callable, TextIO, Dict
from collections import defaultdict
from pathlib import Path


class RunnerStructureDataset(StructureDataset):
  """
  Structure dataset of RuNNer.
  The input structure file contains snapshots of atoms located in the simulation box.
  Each snapshot includes per-atom and collective properties.
  The per-atom properties are element name, coordinates, energy, charge, and force components.
  the collective properties are lattice info, total energy, and total charge.
  """
  # TODO: logging

  def __init__(self, structure_file: Path, transform: Callable = None):   
    """
    Args:
        structure_file (Path): path to the RuNNer structure file.
        transform (Callable, optional): Optional transform to be applied on a structure. Defaults to None.
    """
    self.structure_file = Path(structure_file)
    self.transform = transform
    logger.info(f"Initializing {self.__class__.__name__}(structure_file='{self.structure_file}')")

  def __len__(self) -> int:
    """
    This method opens the structure file and quickly finds the number of structures.

    Returns:
        int: Total number of structures
    """
    n_structures = 0
    with open(str(self.structure_file), "r") as file:
      while self.ignore(file):
        n_structures += 1
    return n_structures

  def __getitem__(self, index) -> Dict:
    """
    Return *it*th structure.
    Multiple-indexing is also supported.

    Args:
        index (_type_): Index of structure.

    Returns:
        Dict: Data structure
    """
    # TODO: index as a list
    # TODO: assert range of index
    return self._read(index)

  def ignore(self, file: TextIO) -> bool:
    """
    This method ignores the next structure.
    It reduces the spent time while reading a range of structures.

    Args:
        file (TextIO): Input structure file handler

    Returns:
        bool: whether ignoring the next structure was successful or not
    """
    # Read next structure
    while True:
      # Read one line from file
      line = file.readline()
      if not line:
        return False
      keyword, tokens = tokenize(line)
      # TODO: check begin keyword
      if keyword == "end": 
        break
    return True

  def read(self, file: TextIO) -> Dict:
    """
    This method reads the next structure.

    Args:
        file (TextIO): Input structure file handler

    Returns:
        Dict: Sample of dataset.
    """
    sample = defaultdict(list)
    # Read next structure
    while True:
      # Read one line from file handler
      line = file.readline()
      if not line:
        return False
      # Read keyword and values
      keyword, tokens = tokenize(line)
      # TODO: check begin keyword
      if keyword == "atom":
        sample["position"].append( [float(t) for t in tokens[:3]] )
        sample["element"].append( tokens[3] )
        sample["charge"].append( float(tokens[4]) )
        sample["energy"].append( float(tokens[5]) )
        sample["force"].append( [float(t) for t in tokens[6:9]] )
      elif keyword == "lattice":
        sample["lattice"].append( [float(t) for t in tokens[:3]] )
      elif keyword == "energy":
        sample["total_energy"].append( float(tokens[0]) )
      elif keyword == "charge":
        sample["total_charge"].append( float(tokens[0]) )
      elif keyword == "comment":
        sample["comment"].append(' '.join(line.split()[1:]) )
      elif keyword == "end": 
        break
    return sample

  def _read(self, index) -> Dict:
    """
    This method reads only the *i*th structure.
    """
    logger.debug(f"Reading structure {index}")
    with open(str(self.structure_file), "r") as file:
      for _ in range(index):
        self.ignore(file) 
      sample = self.read(file)

      if self.transform:
        sample = self.transform(sample)

      return sample



