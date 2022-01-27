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
    self.descriptor = {}   # A dictionary of {element: Descriptor} # TODO: short and long descriptors
    self.model = {}        # A dictionary of {element: Model} # TODO: short and long models

    self._read_config()
    self._construct_descriptor()

  def _read_config(self):
    """
    This method read all NNP configurations from the input file including elements, cutoff type, 
    symmetry functions, neural network, traning parameters, etc. 
    # TODO: read all NNP configuration file.
    # See N2P2 -> https://compphysvienna.github.io/n2p2/topics/keywords.html
    """
    _to_cutoff_type = {  # TODO: poly 3 & 4
        '1': 'hard',
        '2': 'tanhu',
        '3': 'tanh',
        '4': 'exp',
        '5': 'poly1',
        '6': 'poly2',
      }  
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
          try:
            asf_ = (tokens[0], int(tokens[1]), tokens[2]) + tuple([float(t) for t in tokens[3:]])
          except ValueError:
            asf_ = (tokens[0], int(tokens[1]), tokens[2], tokens[3]) + tuple([float(t) for t in tokens[4:]])
          self._config[keyword].append(asf_) 
          # TODO: read angular parameters
        # TODO: asf scaler parameters

    # TODO: add logging
    print("NNP configuration")
    for k, v in self._config.items():
      if isinstance(v, list):
        print(k)
        for i in v:
          print(i)
      else:
          print(f"{k}: {v}")

  def _construct_descriptor(self):
    """
    Construct a descriptor for each element and add the relevant radial and angular symmetry 
    functions from the potential configuration. 
    """
    for element in self._config["elements"]:
      logger.info(f"Instantiating an ASF descriptor for element '{element}'") # TODO: move logging inside ASF method
      self.descriptor[element] = ASF(element)
    for cfg in self._config["symfunction_short"]:
      # logger.info(f"Adding symmetry function: {asf}") # TODO: move logging inside .add() method
      if cfg[1] == 2:
        # TODO: use **kwargs as input argument?
        self.descriptor[cfg[0]].add(
            symmetry_function=G2(r_cutoff=cfg[5], cutoff_type=self._config["cutoff_type"], r_shift=cfg[4], eta=cfg[3]), 
            neighbor_element1=cfg[2]) 

  def train(self, structure_loader: StructureLoader):
    """
    Train the model using the input structure loader.
    """
    # TODO: avoid reading and calculating descriptor multiple times
    # TODO: descriptor element should be the same atom type as the aid
    structures = read_structures(structure_loader, between=(1, 10))
    return self.descriptor["H"](structures[0], aid=5), structures[0].position

     




