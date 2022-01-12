
from .logger import logger
from .structure import Structure
from .loader import StructureLoader


class Potential:
  """
  This class contains all required data and operations to train the ML potential 
  including structures, descriptors, models, etc. 
  TODO: A base class for different types of potentials such as NNP and GAP.
  TODO: split structures from the potential model
  """

  def __init__(self, ):
    self.structures = []

  def read_structures(self, loader: StructureLoader) -> None:
    """
    Read and instantiate structures using the input structure loader.
    """
    # Loop over structure data
    for index, data in enumerate(loader.get_data(), start=1):
      logger.info(f"Reading structure {index}")
      self.structures.append( Structure(data) )      

  def write_structures(self):
    raise NotImplementedError

  @property
  def structures_num(self):
    return len(self.structures)

