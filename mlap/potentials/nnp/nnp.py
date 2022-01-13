
from ...logger import logger
from ...structure import Structure
from ...loader import StructureLoader
from ..base import Potential


class NeuralNetworkPotential (Potential):
  """
  This class contains all required data and operations to train a high-dimensional neural network potential 
  including structures, descriptors, and neural networks. 
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

