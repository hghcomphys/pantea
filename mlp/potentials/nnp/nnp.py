from ...logger import logger
from ...structure import Structure
from ...loaders import StructureLoader, read_structures
from ...descriptors.asf.asf import ASF
from ...descriptors.asf.scaler import AsfScaler
from ...descriptors.asf.radial import G1, G2
from ...descriptors.asf.angular import G3, G9
from ...utils.tokenize import tokenize
from ...utils.batch import create_batch
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
    self._config = None      # A dictionary representation of the NNP configuration file including descriptor and model
    self.descriptor = None   # A dictionary of {element: Descriptor} # TODO: short and long descriptors
    self.scaler = None       # A dictionary of {element: Scaler} # TODO: short and long scalers
    self.model = None        # A dictionary of {element: Model} # TODO: short and long models

    self._read_config()
    self._construct_descriptor()

  def _read_config(self) -> None:
    """
    This method read all NNP configurations from the input file including elements, cutoff type, 
    symmetry functions, neural network, traning parameters, etc. 
    # TODO: read all NNP configuration file.
    # See N2P2 -> https://compphysvienna.github.io/n2p2/topics/keywords.html
    """
    if self._config is not None:
      return
    # Define conversion dictionary
    _to_cutoff_type = {  # TODO: poly 3 & 4
      '1': 'hard',
      '2': 'tanhu',
      '3': 'tanh',
      '4': 'exp',
      '5': 'poly1',
      '6': 'poly2',
      }  
    _to_scaler_type = {   # TODO: center & scaler
      'scale_symmetry_functions': 'scale_center',
      'scale_symmetry_functions_sigma': 'scale_sigma',
    }
    # Read configs from file
    logger.info(f"Reading NNP configuration: file='{self.filename}'")
    self._config = defaultdict(list)
    with open(self.filename, 'r') as file:
      while True:
        # Read the next line
        line = file.readline()
        if not line:
          break
        # Read descriptor parameters
        keyword, tokens = tokenize(line, comment='#')
        if keyword is not None:
          logger.debug(f"keyword='{keyword}', values={tokens}")
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
        # Read symmetry function scaler parameters
        elif keyword == "scale_symmetry_functions":
          self._config["scaler_type"] = _to_scaler_type[keyword]
        elif keyword == "scale_symmetry_functions_sigma":
          self._config["scaler_type"] = _to_scaler_type[keyword]
        elif keyword == "scale_min_short":
          self._config[keyword] = float(tokens[0])
        elif keyword == "scale_max_short":
          self._config[keyword] = float(tokens[0])
        # Read neural network parameters
    logger.info(f"Finished reading NNP configuration")

  def _construct_descriptor(self) -> None:
    """
    Construct a descriptor for each element and add the relevant radial and angular symmetry 
    functions from the potential configuration. 
    TODO: add logging
    """
    if self.descriptor is not None:
      return
    self.descriptor = {}
    self.scaler = {}

    # Instantiate ASF for each element 
    logger.info(f"Elements={self._config['elements']}")
    for element in self._config["elements"]:
      logger.info(f"Creating an ASF descriptor for element '{element}'") # TODO: move logging inside ASF method
      self.descriptor[element] = ASF(element)

    # Add symmetry functions
    logger.info(f"Adding symmetry functions: radial and angular") # TODO: move logging inside .add() method
    for cfg in self._config["symfunction_short"]:
      if cfg[1] == 1:
        # TODO: use **kwargs as input argument?
        self.descriptor[cfg[0]].add(
            symmetry_function = G1(r_cutoff=cfg[5], cutoff_type=self._config["cutoff_type"]), 
            neighbor_element1 = cfg[2]) 
      elif cfg[1] == 2:
        # TODO: use **kwargs as input argument?
        self.descriptor[cfg[0]].add(
            symmetry_function = G2(r_cutoff=cfg[5], cutoff_type=self._config["cutoff_type"], r_shift=cfg[4], eta=cfg[3]), 
            neighbor_element1 = cfg[2]) 
      elif cfg[1] == 3:
        self.descriptor[cfg[0]].add(
            symmetry_function = G3(r_cutoff=cfg[7], cutoff_type=self._config["cutoff_type"], eta=cfg[4], 
              zeta=cfg[6], lambda0=cfg[5], r_shift=0.0), # TODO: add r_shift!
            neighbor_element1 = cfg[2],
            neighbor_element2 = cfg[3]) 
      elif cfg[1] == 9:
        self.descriptor[cfg[0]].add(
            symmetry_function = G9(r_cutoff=cfg[7], cutoff_type=self._config["cutoff_type"], eta=cfg[4], 
              zeta=cfg[6], lambda0=cfg[5], r_shift=0.0), # TODO: add r_shift!
            neighbor_element1 = cfg[2],
            neighbor_element2 = cfg[3]) 
    logger.info("Finished adding symmetry functions")

    # Assign an ASF scaler to each element 
    # TODO: move scaler to the ASF descriptor
    for element in self._config["elements"]:
      logger.info(f"Creating a descriptor scaler for element '{element}'") # TODO: move logging inside scaler class
      self.scaler[element] = AsfScaler()

  def _construct_model(self) -> None:
    """
    Construct a neural network for each element.
    TODO: complete
    """
    if self.model is not None:
      return
    self.model = {}

  def fit_scaler(self, structure_loader: StructureLoader):
    """
    Fit scalers of descriptor for each element based on provided structure loader.
    # TODO: split scaler, define it as separate step in pipeline
    """
    logger.info("Fitting symmetry function scalers...")
    index = 0
    for data in structure_loader.get_data():
      index += 1
      structure = Structure(data)
      for element in self.descriptor.keys():
        aids = structure.select(element).cpu().numpy()
        for batch in create_batch(aids, 10):
          print(f"Structure={index}, element={element}, batch={batch}")
          descriptor = self.descriptor[element](structure, aid=batch) 
          self.scaler[element].fit(descriptor)
          # self.scaler[element].transform(descriptor)
          # break
      # if index >= 2:
      #   break

  def read_scaler(self, filename: str):
    """
    Read scaler parameters.
    No need to fit the scalers in this case. 
    """
    pass
     
  def fit(self, structure_loader: StructureLoader):
    """
    Fit the model using the input structure loader.
    """
    # TODO: avoid reading and calculating descriptor multiple times
    # TODO: descriptor element should be the same atom type as the aid
    # structures = read_structures(structure_loader, between=(1, 10))
    # return self.descriptor["H"](structures[0], aid=0), structures[0].position
    pass


    

    



