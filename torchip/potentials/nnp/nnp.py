from ...logger import logger
from ...structure import Structure
from ...loaders import StructureLoader, read_structures
from ...descriptors.asf.asf import ASF
from ...descriptors.asf.cutoff import CutoffFunction
from ...descriptors.asf.scaler import AsfScaler
from ...descriptors.asf.radial import G1, G2
from ...descriptors.asf.angular import G3, G9
from ...utils.tokenize import tokenize
from ...utils.batch import create_batch
from ...utils.profiler import Profiler
from ...structure.element import ElementMap
from ...config import CFG
from ..base import Potential
from collections import defaultdict, Counter
from typing import Dict, List
from pathlib import Path
import torch
import numpy as np


class NeuralNetworkPotential(Potential):
  """
  This class contains all required data and operations to train a high-dimensional neural network potential 
  including structures, descriptors, and neural networks. 
  TODO: split structures from the potential model
  TODO: implement structure dumper/writer
  TODO: split structure from the potential model (in design)
  """
  def __init__(self, filename: Path) -> None:
    self.filename = Path(filename)
    self._settings = None    # A dictionary representation of the NNP settgins including descriptor, scaler, and model
    self.descriptor = None   # A dictionary of {element: Descriptor}   # TODO: short and long descriptors
    self.scaler = None       # A dictionary of {element: Scaler}       # TODO: short and long scalers
    self.model = None        # A dictionary of {element: Model}        # TODO: short and long models
    logger.info(f"Initializing {self.__class__.__name__}") # TODO: define __repr__

    if self._settings is None:
      self._read_settings_file()
      self._construct_descriptor()
      self._construct_scaler()
      self._construct_model()

  def _read_settings_file(self) -> None:
    """
    This method reads all NNP settings from the file including elements, cutoff type, 
    symmetry functions, neural network, traning parameters, etc. 
    See N2P2 -> https://compphysvienna.github.io/n2p2/topics/keywords.html
    """
    # Map cutoff type
    _to_cutoff_type = {  # TODO: poly 3 & 4
      '0': 'hard',
      '1': 'cos',
      '2': 'tanhu',
      '3': 'tanh',
      '4': 'exp',
      '5': 'poly1',
      '6': 'poly2',
    }  
    # Map scaler type
    _to_scaler_type = {   # TODO: center & scaler
      'scale_symmetry_functions': 'scale_center',
      'scale_symmetry_functions_sigma': 'scale_sigma',
    }
    # Read setting file
    logger.info(f"Reading NNP settings file:'{self.filename}'")
    self._settings = defaultdict(list)
    with open(str(self.filename), 'r') as file:
      while True:
        # Read the next line
        line = file.readline()
        if not line:
          break
        # Read descriptor parameters
        keyword, tokens = tokenize(line, comment='#')
        if keyword is not None:
          logger.debug(f"keyword:'{keyword}', values:{tokens}")
        if keyword == "number_of_elements":
          self._settings[keyword] = int(tokens[0])
        elif keyword == "elements":
          self._settings[keyword] = sorted(set([t for t in tokens]), key=ElementMap.get_atomic_number)
        elif keyword == "cutoff_type":
          self._settings[keyword] = _to_cutoff_type[tokens[0]]
        elif keyword == "symfunction_short":
          try:
            asf_ = (tokens[0], int(tokens[1]), tokens[2]) + tuple([float(t) for t in tokens[3:]])
          except ValueError:
            asf_ = (tokens[0], int(tokens[1]), tokens[2], tokens[3]) + tuple([float(t) for t in tokens[4:]])
          self._settings[keyword].append(asf_) 
        # Read symmetry function scaler parameters
        elif keyword == "scale_symmetry_functions":
          self._settings["scale_type"] = _to_scaler_type[keyword]
        elif keyword == "scale_symmetry_functions_sigma":
          self._settings["scale_type"] = _to_scaler_type[keyword]
        elif keyword == "scale_min_short":
          self._settings[keyword] = float(tokens[0])
        elif keyword == "scale_max_short":
          self._settings[keyword] = float(tokens[0])
        # Read neural network parameters

  def _construct_descriptor(self) -> None:
    """
    Construct a descriptor for each element and add the relevant radial and angular symmetry 
    functions from the a dictionary representation of potential settings.
    TODO: add logging
    """
    self.descriptor = {}

    # Elements
    logger.info(f"Number of elements: {len(self._settings['elements'])}")
    for element in self._settings["elements"]:
      logger.info(f"Element '{element}' ({ElementMap.get_atomic_number(element):<3})") 

    # Instantiate ASF for each element 
    logger.info(f"Creating ASF descriptors")
    for element in self._settings["elements"]:
      self.descriptor[element] = ASF(element)

    # Add symmetry functions
    logger.info(f"Adding symmetry functions: radial and angular") # TODO: move logging inside .add() method
    for cfg in self._settings["symfunction_short"]:
      if cfg[1] == 1:
        self.descriptor[cfg[0]].add(
          symmetry_function = G1(CutoffFunction(r_cutoff=cfg[5], cutoff_type=self._settings["cutoff_type"])), 
          neighbor_element1 = cfg[2]
        ) 
      elif cfg[1] == 2:
        self.descriptor[cfg[0]].add(
          symmetry_function = G2(CutoffFunction(r_cutoff=cfg[5], cutoff_type=self._settings["cutoff_type"]), 
            eta=cfg[3], r_shift=cfg[4]), 
          neighbor_element1 = cfg[2]
        )
      elif cfg[1] == 3:
        self.descriptor[cfg[0]].add(
          symmetry_function = G3(CutoffFunction(r_cutoff=cfg[7], cutoff_type=self._settings["cutoff_type"]), 
            eta=cfg[4], zeta=cfg[6],  lambda0=cfg[5], r_shift=0.0), # TODO: add r_shift!
          neighbor_element1 = cfg[2],
          neighbor_element2 = cfg[3]
        ) 
      elif cfg[1] == 9:
        self.descriptor[cfg[0]].add(
          symmetry_function = G9(CutoffFunction(r_cutoff=cfg[7], cutoff_type=self._settings["cutoff_type"]), 
            eta=cfg[4], zeta=cfg[6], lambda0=cfg[5], r_shift=0.0), # TODO: add r_shift!
          neighbor_element1 = cfg[2],
          neighbor_element2 = cfg[3]
        ) 

  def _construct_scaler(self) -> None:
    """
    Construct a descriptor for each element using a dictionary representation of NNP settings.
    """
    self.scaler = {}

    # Prepare scaler input argument if exist in settings
    kwargs = { first: self._settings[second] \
      for first, second in { 
          'scale_type': 'scale_type', 
          'scale_min': 'scale_min_short',
          'scale_max': 'scale_max_short',
        }.items() if second in self._settings
    }
    logger.debug(f"Preparing ASF scaler kwargs={kwargs}")

    # Assign an ASF scaler to each element
    logger.info(f"Creating ASF descriptor scalers")
    for element in self._settings["elements"]:
      self.scaler[element] = AsfScaler(**kwargs) 

  def _construct_model(self) -> None:
    """
    Construct a neural network for each element using a dictionary representation of NNP settings.
    """
    self.model = {}
    # TODO: complete

  @Profiler.profile
  def fit_scaler(self, structure_loader: StructureLoader, filename: Path = None) -> None:
    """
    Fit scalers of descriptor for each element using the provided input structure loader.
    # TODO: split scaler, define it as separate step in pipeline
    """
    logger.info("Fitting symmetry function scalers")
    for index, data in enumerate(structure_loader.get_data(), start=1):
      structure = Structure(data, 
                            r_cutoff=self.r_cutoff,  # global cutoff radius (maximum) 
                            requires_grad=False)     # No need to track graph history (gradient) 
      for element, scaler in self.scaler.items():
        aids = structure.select(element).detach()
        for batch in create_batch(aids, 10):
          logger.debug(f"Structure={index}, element='{element}', batch={batch}")
          descriptor = self.descriptor[element](structure, aid=batch) 
          scaler.fit(descriptor)

    # Save scaler data into file  
    if filename is not None:
      logger.info(f"Saving scaler data file='{filename}'")
      with open(str(Path(filename)), "w") as file:
        file.write(f"# Symmetry function scaling data\n")
        file.write(f"# {'Element':<10s} {'Min':<23s} {'Max':<23s} {'Mean':<23s} {'Sigma':<23s}\n")
        for element, scaler in self.scaler.items():     
          for i in range(scaler.dimension):
            file.write(f"  {element:<10s} ")
            file.write(f"{scaler.min[i]:<23.15E} {scaler.max[i]:<23.15E} {scaler.mean[i]:<23.15E} {scaler.sigma[i]:<23.15E}\n")

  @Profiler.profile
  def read_scaler(self, filename: Path) -> None:
    """
    Read scaler parameters.
    No need to fit the scalers in this case. 
    """
    logger.info(f"Reading scaler data file='{filename}'")
    data = np.loadtxt(filename, usecols=(1, 2, 3, 4))
    element_count = Counter(np.loadtxt(filename, usecols=(0), dtype=str))
    index = 0
    for element, count in element_count.items():
      scaler= self.scaler[element]
      data_ = data[index:index+count, :]
      scaler.sample = 1
      scaler.dimension = data.shape[1]
      scaler.min   = torch.tensor(data_[:, 0], device=CFG["device"]) # TODO: dtype?
      scaler.max   = torch.tensor(data_[:, 1], device=CFG["device"])
      scaler.mean  = torch.tensor(data_[:, 2], device=CFG["device"])
      scaler.sigma = torch.tensor(data_[:, 3], device=CFG["device"])
      index += count

  def fit_model(self, structure_loader: StructureLoader) -> None:
    """
    Fit the model using the input structure loader.
    """
    # TODO: avoid reading and calculating descriptor multiple times
    # TODO: descriptor element should be the same atom type as the aid
    # structures = read_structures(structure_loader, between=(1, 10))
    # return self.descriptor["H"](structures[0], aid=0), structures[0].position
    pass

  def fit(self)  -> None:
    """
    Fit descriptor and model (if needed). 
    """
    pass

  @property
  def elements(self) -> List[str]:
    return self._settings['elements']

  @property
  def r_cutoff(self) -> float:
    """
    Return the maximum cutoff radius of all elemental descriptors.
    """
    return max([dsc.r_cutoff for dsc in self.descriptor.values()])


    

    



