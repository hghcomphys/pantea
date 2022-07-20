from ..logger import logger
from ..structure import Structure
from ..datasets.transformer import ToStructure
from ..datasets.runner import RunnerStructureDataset
from ..descriptors.asf.asf import ASF
from ..descriptors.asf.cutoff import CutoffFunction
from ..descriptors.scaler import DescriptorScaler
from ..descriptors.asf.radial import G1, G2
from ..descriptors.asf.angular import G3, G9
from ..models.nn import NeuralNetworkModel
from ..utils.tokenize import tokenize
from ..utils.batch import create_batch
from ..utils.profiler import Profiler
from ..structure.element import ElementMap
from ..config import dtype, device
from .base import Potential
from .trainer import NeuralNetworkPotentialTrainer
from collections import defaultdict
from typing import List, Dict
from torch.utils.data import DataLoader as TorchDataLoader
from pathlib import Path
from torch import Tensor
from torch import nn
import torch


class NeuralNetworkPotential(Potential):
  """
  A suitcase class of high-dimensional neural network potential (HDNNP). 
  It contains all the required descriptors, scalers, neural networks, and a trainer to fit the potential using 
  provided structure data and a potential setting file. 
  """
  # TODO: split structures from the potential model
  # TODO: implement structure dumper/writer
  # TODO: split structure from the potential model (in design)

  # Default settings
  _default_settings = {
    "symfunction_short": [],
    "epochs": 1,
    "updater_type": 0,
    "gradient_type": 1, 
  }
  # Map cutoff type
  _map_cutoff_type = {  # TODO: poly 3 & 4
    '0': 'hard',
    '1': 'cos',
    '2': 'tanhu',
    '3': 'tanh',
    '4': 'exp',
    '5': 'poly1',
    '6': 'poly2',
  }  
  # Map scaler type
  _map_scaler_type = {  
    'center_symmetry_functions': 'center',
    'scale_symmetry_functions': 'scale',
    'scale_center_symmetry_functions': 'scale_center',
    'scale_center_symmetry_functions_sigma': 'scale_center_sigma',
  }
  # saving formats
  _scaler_save_format = "scaling.{:03d}.data"
  _model_save_format = "weights.{:03d}.zip"
 
  def __init__(self, potfile: Path) -> None:
    """
    Initialize a HDNNP potential instance by reading the potential file and 
    creating descriptors, scalers, models, and trainer.

    Args:
        potfile (Path): A file path to potential file.
    """

    # Initialization
    self.potfile = Path(potfile)
    self._settings = None    # A dictionary representation of the NNP settgins including descriptor, scaler, and model
    self.descriptor = None   # A dictionary of {element: Descriptor}   # TODO: short and long descriptors
    self.scaler = None       # A dictionary of {element: Scaler}       # TODO: short and long scalers
    self.model = None        # A dictionary of {element: Model}        # TODO: short and long models
    self.trainer = None

    logger.debug(f"Initializing {self.__class__.__name__}(potfile={self.potfile})")
    self._read_settings()
    self._init_descriptor()
    self._init_scaler()
    self._init_model()
    self._init_trainer()

  def _read_settings(self) -> None:
    """
    This method reads all NNP settings from the file including elements, cutoff type, 
    symmetry functions, neural network, training parameters, etc. 
    See N2P2 -> https://compphysvienna.github.io/n2p2/topics/keywords.html
    """
    self._settings = defaultdict(None)
    self._settings.update(self._default_settings)

    # Read settings from file
    logger.debug(f"Reading the HDNNP potential file:'{self.potfile}'")
    with open(str(self.potfile), 'r') as file:

      while True:
        # Read the next line
        line = file.readline()
        if not line:
          break
        
        # Read keyword and values
        keyword, tokens = tokenize(line, comment='#')
        if keyword is not None:
          logger.debug(f"keyword:'{keyword}', values:{tokens}")
        
        # General settings  
        if keyword == "number_of_elements":
          self._settings[keyword] = int(tokens[0])
        elif keyword == "elements":
          self._settings[keyword] = sorted(set([t for t in tokens]), key=ElementMap.get_atomic_number)
        elif keyword == "cutoff_type":
          self._settings[keyword] = self._map_cutoff_type[tokens[0]]
        elif keyword == "symfunction_short":
          try:
            asf_ = (tokens[0], int(tokens[1]), tokens[2]) + tuple([float(t) for t in tokens[3:]])
          except ValueError:
            asf_ = (tokens[0], int(tokens[1]), tokens[2], tokens[3]) + tuple([float(t) for t in tokens[4:]])
          self._settings[keyword].append(asf_)     
        
        # Symmetry function settings
        elif keyword == "center_symmetry_functions":
          self._settings["scale_type"] = self._map_scaler_type[keyword]
        elif keyword == "scale_symmetry_functions":
          self._settings["scale_type"] = self._map_scaler_type[keyword]
        elif keyword == "scale_center_symmetry_functions":
          self._settings["scale_type"] = self._map_scaler_type[keyword]
        elif keyword == "scale_center_symmetry_functions_sigma":
          self._settings["scale_type"] = self._map_scaler_type[keyword]
        elif keyword == "scale_min_short":
          self._settings[keyword] = float(tokens[0])
        elif keyword == "scale_max_short":
          self._settings[keyword] = float(tokens[0])

        # Trainer settings
        elif keyword == "epochs":
          self._settings[keyword] = int(tokens[0])
        elif keyword == "updater_type":
          self._settings[keyword] = int(tokens[0])
        elif keyword == "gradient_type":
          self._settings[keyword] = int(tokens[0])
        elif keyword == "gradient_eta":
          self._settings[keyword] = float(tokens[0])
        elif keyword == "gradient_adam_eta":
          self._settings[keyword] = float(tokens[0])
        elif keyword == "gradient_adam_beta1":
          self._settings[keyword] = float(tokens[0])
        elif keyword == "gradient_adam_beta2":
          self._settings[keyword] = float(tokens[0])
        elif keyword == "gradient_adam_epsilon":
          self._settings[keyword] = float(tokens[0])
        
  def _init_descriptor(self) -> None:
    """
    Initialize a **descriptor** for each element and add the relevant radial and angular 
    symmetry functions from the potential settings.
    """
    # TODO: add logging 
    self.descriptor = {}

    # Elements
    logger.info(f"Number of elements: {len(self._settings['elements'])}")
    for element in self._settings["elements"]:
      logger.info(f"Element: {element} ({ElementMap.get_atomic_number(element)})") 

    # Instantiate ASF for each element 
    logger.debug(f"Creating ASF descriptors")
    for element in self._settings["elements"]:
      self.descriptor[element] = ASF(element)

    # Add symmetry functions
    logger.debug(f"Registering symmetry functions (radial and angular)") # TODO: move logging inside .add() method
    for cfg in self._settings["symfunction_short"]:
      if cfg[1] == 1:
        self.descriptor[cfg[0]].register(
          symmetry_function = G1(CutoffFunction(r_cutoff=cfg[5], cutoff_type=self._settings["cutoff_type"])), 
          neighbor_element1 = cfg[2]
        ) 
      elif cfg[1] == 2:
        self.descriptor[cfg[0]].register(
          symmetry_function = G2(CutoffFunction(r_cutoff=cfg[5], cutoff_type=self._settings["cutoff_type"]), 
            eta=cfg[3], r_shift=cfg[4]), 
          neighbor_element1 = cfg[2]
        )
      elif cfg[1] == 3:
        self.descriptor[cfg[0]].register(
          symmetry_function = G3(CutoffFunction(r_cutoff=cfg[7], cutoff_type=self._settings["cutoff_type"]), 
            eta=cfg[4], zeta=cfg[6],  lambda0=cfg[5], r_shift=0.0), # TODO: add r_shift!
          neighbor_element1 = cfg[2],
          neighbor_element2 = cfg[3]
        ) 
      elif cfg[1] == 9:
        self.descriptor[cfg[0]].register(
          symmetry_function = G9(CutoffFunction(r_cutoff=cfg[7], cutoff_type=self._settings["cutoff_type"]), 
            eta=cfg[4], zeta=cfg[6], lambda0=cfg[5], r_shift=0.0), # TODO: add r_shift!
          neighbor_element1 = cfg[2],
          neighbor_element2 = cfg[3]
        ) 

  def _init_scaler(self) -> None:
    """
    Initialize a descriptor scaler for each element from the potential settings.
    """
    self.scaler = {}

    # Prepare scaler input argument if exist in settings
    scaler_kwargs = { first: self._settings[second] \
      for first, second in { 
          'scale_type': 'scale_type', 
          'scale_min': 'scale_min_short',
          'scale_max': 'scale_max_short',
        }.items() if second in self._settings
    }
    logger.debug(f"Preparing ASF scaler kwargs={scaler_kwargs}")

    # Assign an ASF scaler to each element
    logger.debug(f"Creating descriptor scalers")
    for element in self._settings["elements"]:
      self.scaler[element] = DescriptorScaler(**scaler_kwargs) 

  def _init_model(self) -> None:
    """
    Initialize a neural network for each element using a dictionary representation of potential settings.
    """
    self.model = {}
    # Instantiate neural network model for each element 
    logger.debug(f"Creating neural network models")
    for element in self._settings["elements"]:
      input_size = self.descriptor[element].n_descriptor
      self.model[element] = NeuralNetworkModel(input_size, hidden_layers=((3, 't'), (3, 't')), output_layer=(1, 'l'))
      self.model[element].to(device.DEVICE)
      # TODO: add element argument
      # TODO: read layers from the settings

  def _init_trainer(self) -> None:
    """
    This method initializes a trainer instance including optimizer, loss function, criterion, etc.
    The trainer is used for fitting the energy models. 
    """
    kwargs = {}
    if self._settings["updater_type"] == 0:  # Gradient Descent
      if self._settings["gradient_type"] == 1:  # Adam
        kwargs["criterion"] = nn.MSELoss()
        kwargs["learning_rate"] = self._settings["gradient_adam_eta"] # TODO: defining learning_rate?
        kwargs["optimizer_func"] = torch.optim.Adam
        kwargs["optimizer_func_kwargs"] = {
          "lr": self._settings["gradient_adam_eta"],
          "betas": (self._settings["gradient_adam_beta1"], self._settings["gradient_adam_beta2"]),
          "eps": self._settings["gradient_adam_epsilon"],
        }
      elif self._settings["gradient_type"] == 0:  # Fixed Step  
        logger.error("Gradient descent type fixed step is not implemented yet", 
                      exception=NotImplementedError)

    # Create trainer instance
    logger.debug(f"Preparing trainer kwargs={kwargs}")
    logger.debug(f"Creating NNP trainer")
    self.trainer = NeuralNetworkPotentialTrainer(self, **kwargs)

  @Profiler.profile
  def fit_scaler(self, dataset: RunnerStructureDataset, **kwargs) -> None:
    """
    Fit scaler parameters for each element using the input structure data.

    Args:
        structures (RunnerStructureDataset): Structure dataset
    """    
    # Set parameters
    save_scaler = kwargs.get("save_scaler", True)
    batch_size = kwargs.get("batch_size", 4)  # batch of atoms in each structure

    # Prepare structure dataset and loader (for fitting scaler)
    dataset_ = dataset.copy() # because of using a new transformer (no structure data will be copied)
    dataset_.transform = ToStructure(r_cutoff=self.r_cutoff, requires_grad=False)  
    loader = TorchDataLoader(dataset_, collate_fn=lambda batch: batch)

    logger.info("Fitting descriptor scalers")
    for index, batch in enumerate(loader): 
      # TODO: spawn processes
      structure = batch[0] 
      for element in structure.elements:
        aids = structure.select(element).detach()
        for aids_batch in create_batch(aids, batch_size):
          logger.debug(f"Calculating descriptor for Structure {index} (element='{element}', aids={aids_batch})")
          x = self.descriptor[element](structure, aids_batch) # kernel
          self.scaler[element].fit(x)
    logger.debug("Finished scaler fitting.")

    # Save scaler data into file  
    if save_scaler:
      self.save_scaler()

  def save_scaler(self):
    """
    This method saves scaler parameters for each element into separate files. 
    """
    # Save scaler parameters for each element separately
    for element in self.elements:
      atomic_number = ElementMap.get_atomic_number(element)
      scaler_fn = Path(self.potfile.parent, self._scaler_save_format.format(atomic_number)) 
      logger.info(f"Saving scaler parameters for element ({element}): {scaler_fn.name}")
      self.scaler[element].save(scaler_fn)
    # # Save scaler parameters for all element into a single file
    # scaler_fn = Path(self.potfile.parent, self.scaler_save_format) 
    # logger.info(f"Saving scaler parameters into '{scaler_fn}'")
    # with open(str(scaler_fn), "w") as file:
    #   file.write(f"# Symmetry function scaling data\n")
    #   file.write(f"# {'Element':<10s} {'Min':<23s} {'Max':<23s} {'Mean':<23s} {'Sigma':<23s}\n")
    #   for element, scaler in self.scaler.items():     
    #     for i in range(scaler.dimension):
    #       file.write(f"  {element:<10s} ")
    #       file.write(f"{scaler.min[i]:<23.15E} {scaler.max[i]:<23.15E} {scaler.mean[i]:<23.15E} {scaler.sigma[i]:<23.15E}\n")

  @Profiler.profile
  def load_scaler(self) -> None:
    """
    This method loads scaler parameters of each element from separate files.
    This save computational time as the would be no need to fit the scalers each time. 
    """
    # Load scaler parameters for each element separately
    for element in self.elements:
      logger.debug(f"Loading scaler parameters for element: {element}")
      atomic_number = ElementMap.get_atomic_number(element)
      scaler_fn = Path(self.potfile.parent, self._scaler_save_format.format(atomic_number)) 
      self.scaler[element].load(scaler_fn)
    # # Load scaler parameters for all element into a single file
    # scaler_fn = Path(self.potfile.parent, self.scaler_save_format) 
    # logger.info(f"Loading scaler parameters from '{scaler_fn}'")
    # data = np.loadtxt(scaler_fn, usecols=(1, 2, 3, 4))
    # element_count = Counter(np.loadtxt(scaler_fn, usecols=(0), dtype=str))
    # index = 0
    # for element, count in element_count.items():
    #   scaler= self.scaler[element]
    #   data_ = data[index:index+count, :]
    #   scaler.sample = 1
    #   scaler.dimension = data.shape[1]
    #   scaler.min   = torch.tensor(data_[:, 0], device=CFG["device"]) # TODO: dtype?
    #   scaler.max   = torch.tensor(data_[:, 1], device=CFG["device"])
    #   scaler.mean  = torch.tensor(data_[:, 2], device=CFG["device"])
    #   scaler.sigma = torch.tensor(data_[:, 3], device=CFG["device"])
    #   index += count

  @Profiler.profile
  def fit_model(self, dataset: RunnerStructureDataset, **kwargs) -> Dict:
    """
    Fit energy model for all elements using the input structure loader.
    """
    # TODO: avoid reading and calculating descriptor multiple times
    # TODO: descriptor element should be the same atom type as the aid
    # TODO: define a dataloader specific to energy and force data (shuffle, train & test split)
    # TODO: add validation output (MSE separate for force and energy)
    return self.trainer.fit(dataset, **kwargs)

  def save_model(self):
    """
    Save model weights separately for all elements.
    """
    for element in self.elements:
      logger.debug(f"Saving model weights for element: {element}")   
      atomic_number = ElementMap.get_atomic_number(element)
      model_fn = Path(self.potfile.parent, self._model_save_format.format(atomic_number))
      self.model[element].save(model_fn)

  @Profiler.profile
  def load_model(self):
    """
    Load model weights separately for all elements.
    """
    for element in self.elements:
        logger.debug(f"Loading model weights for element: {element}")
        atomic_number = ElementMap.get_atomic_number(element)
        model_fn = Path(self.potfile.parent, self._model_save_format.format(atomic_number))
        self.model[element].load(model_fn)

  def fit(self)  -> None:
    """
    This method provides a user-friendly interface to fit both descriptor and model in one step. 
    """
    pass

  def __call__(self, structure: Structure) -> Tensor:
    """
    Calculate the total energy of the input structure.
    """
    structure_ = structure.copy(r_cutoff=self.r_cutoff)

    # Set models in evaluation status
    for element in self.elements:
      self.model[element].eval()

    # Loop over elements
    energy = 0.0
    for element in self.elements:
      aids = structure_.select(element).detach()
      x = self.descriptor[element](structure_, aid=aids)
      x = self.scaler[element](x)
      x = self.model[element](x)
      x = torch.sum(x, dim=0)
      energy = energy + x

    return energy

  @property
  def elements(self) -> List[str]:
    return self._settings['elements']

  @property
  def r_cutoff(self) -> float:
    """
    Return the maximum cutoff radius found between all descriptors.
    """
    return max([dsc.r_cutoff for dsc in self.descriptor.values()])

