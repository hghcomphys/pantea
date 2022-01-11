
from .logger import logger
from .structure import Structure
from .loader import StructureLoader


class System:
  """
  This class contains all required info train the potential including structures, descriptors, potential models, etc. 
  """

  def __init__(self):
    self.structures = None
    self.numOfStructures = None

  def read_structures(self, loader: StructureLoader) -> None:
    """
    Read all structures.
    """
    # Initialize
    self.structures = []
    self.numOfStructures = 0

    for data in loader.get_data():
      # Append to the list of structures
      logger.info(f"Reading structure {self.numOfStructures + 1}")
      self.structures.append( Structure(data) )
      self.numOfStructures += 1        

  def write_structures(self):
    raise NotImplementedError

