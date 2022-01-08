
from .logger import logger
from .structure import Structure


class System:
  """
  This class contains all required info train the potential including structures, descriptors, potential models, etc. 
  """

  def __init__(self):
    self.structures = None
    self.numOfStructures = None

  def read_structures(self, filename: str) -> None:
    """
    Read all structures.
    """
    # Initialize
    self.structures = []
    self.numOfStructures = 0

    # Open structure file
    logger.info(f"Opening structure file {filename}")
    with open(filename, "r") as file:
      while True:
        # TODO: generator design
        structure = Structure()
        # Read next structure
        if ( not structure.read(file) ):
          break
        # Append to the list of structures
        logger.info(f"Reading structure {self.numOfStructures + 1}")
        self.structures.append( structure )
        self.numOfStructures += 1

  def write_structures(self):
    pass

