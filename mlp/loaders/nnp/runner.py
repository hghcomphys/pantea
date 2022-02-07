from ...logger import logger
from ..base import StructureLoader
from ...utils.tokenize import tokenize
from typing import Tuple, List, TextIO, Dict
from collections import defaultdict
from pathlib import Path


class RunnerStructureLoader(StructureLoader):
  """
  A derived class of structure loader for the NNP (RuNNer) file format.
  TODO: logging
  TODO: define a derived structure loader class specific to NNP and leave the base class here 
  """

  def __init__(self, filename: Path) -> None:
    self.filename = Path(filename)
    self._data = None
    self._ignore_next = False
    logger.info(f"Initializing {self.__class__.__name__}: file='{self.filename}'")

  def get_data(self) -> Dict[str, List]:
    """
    A generator method which returns a each snapshot of atomic data structure as a dictionary.
    The output data can be used to instantiate, for example, Structure class. 
    """
    with open(str(self.filename), "r") as file:
      try:
        while ( self.read(file) if not self._ignore_next else self.ignore(file) ):
          yield self._data
      except AttributeError as err:
        logger.warning(f"It seems that {self.__class__.__name__} has no 'ignore()' method defined")
        while self.read(file):
          yield self._data
    # Clean up
    self._data = None
    self._ignore_next = False

  def read(self, file: TextIO) -> bool:
    """
    This method reads the next structure from the given input file.
    """
    self._data = defaultdict(list)
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
      elif keyword == "end": 
        break
    return True

  def ignore(self, file: TextIO) -> bool:
    """
    This method ignores the next structure.
    It reduces time spending on reading a range of structures and not all of them.
    This is an optional method that can be define in a derived structure loader to reach a better I/O performance.
    """
    self._data = None
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
    self._ignore_next = False
    return True

  def ignore_next(self):
    """
    Set the internal variable true.
    """
    self._ignore_next = True

