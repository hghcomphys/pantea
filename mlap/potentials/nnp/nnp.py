
from calendar import c
from ...logger import logger
from ...structure import Structure
from ...loaders import StructureLoader, read_structures
from ...descriptors.asf.asf import ASF
from ...descriptors.asf.radial import G1, G2
from ...utils.tokenize import tokenize
from ..base import Potential
from collections import defaultdict


class NeuralNetworkPotential(Potential):
  """
  This class contains all required data and operations to train a high-dimensional neural network potential 
  including structures, descriptors, and neural networks. 
  TODO: split structures from the potential model
  TODO: implement structure dumper/writer
  """
  def __init__(self, filename: str) -> None:
    self.filename = filename
    self._config = None
    self.descriptor = {}  # A dictionary of {element: Descriptor} # TODO: short and long descriptor
    self.model = {}        # A dictionary of {element: Model}

    self._read_config()
    self._construct_descriptors()

  def _read_config(self):
    """
    This method read all NNP configurations from the input file including elements, cutoff type, 
    symmetry functions, neural network, traning parameters, etc. 
    # TODO: read all NNP configuration file.
    """
    _to_cutoff_type = {'2': 'tanh'}  # TODO: complete the list, move to ASF class?
    _to_asf_type  = {'2': 'radial', '3': 'angular'}  # TODO: complete the list, move to ASF class?
    self._config = defaultdict(list)
    with open(self.filename, 'r') as file:
      while True:
        # Read the next line
        line = file.readline()
        if not line:
          break
        # Read keyword and values
        keyword, tokens = tokenize(line, comment='#')
        if keyword == "number_of_elements":
          self._config[keyword] = int(tokens[0])
        elif keyword == "elements":
          self._config[keyword] = tuple(set([t for t in tokens]))
        elif keyword == "cutoff_type":
          self._config[keyword] = _to_cutoff_type[tokens[0]]
        elif keyword == "symfunction_short":
          asf_type = _to_asf_type[tokens[1]] # TODO: there are multiple type of SF terms both for radial and angular terms
          if asf_type == 'radial':
            self._config[keyword].append((tokens[0],                   # central element
                                    int(tokens[1]),               # asf type
                                    tokens[2],                    # neighbor element1
                                    float(tokens[3]),             # eta
                                    float(tokens[4]),             # r-shift
                                    float(tokens[5])) )           # r_cutoff
          # TODO: read angular parameters
        # TODO: asf scaler parameters

    # TODO: add logging
    print("NNP configuration")
    for k, v in self._config.items():
      print(f"{k}: {v}")

  def _construct_descriptors(self):
    """
    Construct a descriptor for each element and add the relevant radial and angular symmetry 
    functions from the potential configuration. 
    """
    for element in self._config["elements"]:
      logger.info(f"Instantiating an ASF descriptor for element '{element}'") # TODO: move logging inside ASF method
      self.descriptor[element] = ASF(element)
    for asf in self._config["symfunction_short"]:
      # logger.info(f"Adding symmetry function: {asf}") # TODO: move logging inside .add() method
      if asf[1] == 2:
        # TODO: use **kwargs as input argument
        self.descriptor[asf[0]].add(
            symmetry_function=G2(r_cutoff=asf[5], cutoff_type=self._config["cutoff_type"], r_shift=asf[4], eta=asf[3]), 
            neighbor_element1=asf[2]) 

  def train(self, structure_loader: StructureLoader):
    """
    Train the model using the input structure loader.
    """
    # TODO: avoid reading and calculating descriptor multiple times
    # TODO: descriptor element should be the same atom type as the aid
    structures = read_structures(structure_loader, between=(1, 10))
    return self.descriptor["H"](structures[0], aid=2)

     




