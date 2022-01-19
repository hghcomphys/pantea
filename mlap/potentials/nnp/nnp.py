
from ...logger import logger
from ...structure import Structure
from ...loader import StructureLoader
from ...descriptors.asf.asf import ASF
from ...descriptors.asf.radial import G1, G2
from ...utils import read_structures
from ..base import Potential


class NeuralNetworkPotential(Potential):
  """
  This class contains all required data and operations to train a high-dimensional neural network potential 
  including structures, descriptors, and neural networks. 
  TODO: split structures from the potential model
  TODO: implement structure dumper/writer
  """
  def __init__(self, filename: str) -> None:
    self.filename = filename

    elements = ["H", "O"]
    self.descriptors = { element: ASF(element) for element in elements }

    r_cutoff, cutoff_type = 12.0, "tanh"
    for element in elements:
      self.descriptors[element].add(G1(r_cutoff, cutoff_type), "H")
      self.descriptors[element].add(G2(r_cutoff, cutoff_type, r_shift=0.0, eta=0.3), "H")
      self.descriptors[element].add(G2(r_cutoff, cutoff_type, r_shift=0.0, eta=0.3), "O")

  def train(self, structure_loader: StructureLoader):
    """
    Train the model using the input structure loader.
    """
    # TODO: avoid reading and calculating descriptor multiple times
    # TODO: descriptor element should be the same atom type as the aid
    structures = read_structures(structure_loader)
    return self.descriptors["H"](structures[0], aid=2)

     




