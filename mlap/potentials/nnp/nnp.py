
from calendar import c
from ...logger import logger
from ...structure import Structure
from ...loaders import StructureLoader, read_structures
from ...descriptors.asf.asf import ASF
from ...descriptors.asf.radial import G1, G2
from ...utils.tokenize import tokenize
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
    self.descriptors = {}  # A dictionary of {element: Descriptor}
    self.model = {}        # A dictionary of {element: Model}

    self.read()

  def read(self):
    """
    This method read atomic symmetry function from input file.
    # TODO: it should read all NNP configuration file.
    """
    _to_cutoff_type = {'2': 'tanh'}  # TODO: complete the list
    _config = {}
    with open(self.filename, 'r') as file:
      while True:
        # Read the next line
        line = file.readline()
        if not line:
          break
        # Read keyword and values
        keyword, tokens = tokenize(line, comment='#')
        if keyword == "number_of_elements":
          _config[keyword] = int(tokens[0])
        elif keyword == "elements":
          _config[keyword] = [t for t in tokens]
        elif keyword == "cutoff_type":
          _config[keyword] = _to_cutoff_type[tokens[0]]
        # TODO: asf scaler parameters

    print("NNP configuration")
    for k, v in _config.items():
      print(f"{k}: {v}")

    # elements = ["H", "O"]
    # self.descriptors = { element: ASF(element) for element in elements }

    # r_cutoff, cutoff_type = 12.0, "tanh"
    # for element in elements:
    #   self.descriptors[element].add(G1(r_cutoff, cutoff_type), "H")
    #   self.descriptors[element].add(G2(r_cutoff, cutoff_type, r_shift=0.0, eta=0.3), "H")
    #   self.descriptors[element].add(G2(r_cutoff, cutoff_type, r_shift=0.0, eta=0.3), "O")

  def train(self, structure_loader: StructureLoader):
    """
    Train the model using the input structure loader.
    """
    # TODO: avoid reading and calculating descriptor multiple times
    # TODO: descriptor element should be the same atom type as the aid
    structures = read_structures(structure_loader)
    return self.descriptors["H"](structures[0], aid=2)

     




