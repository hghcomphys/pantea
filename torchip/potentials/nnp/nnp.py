from ...logger import logger
from ...structure import Structure
from ...loaders import StructureLoader
from ...descriptors.asf.asf import ASF
from ...descriptors.asf.cutoff import CutoffFunction
from ...descriptors.scaler import DescriptorScaler
from ...descriptors.asf.radial import G1, G2
from ...descriptors.asf.angular import G3, G9
from ...models.nn.nn import NeuralNetworkModel
from ...utils.tokenize import tokenize
from ...utils.batch import create_batch
from ...utils.profiler import Profiler
from ...structure.element import ElementMap
from ...config import CFG
from ..base import Potential
from .trainer import NeuralNetworkPotentialTrainer
from collections import defaultdict
from typing import List, Dict
from pathlib import Path
from torch import Tensor
from torch import nn
import torch


class NeuralNetworkPotential(Potential):
  """
  This class contains all required data and operations to train a high-dimensional neural network potential 
  including structures, descriptors, and neural networks. 
  TODO: split structures from the potential model
  TODO: implement structure dumper/writer
  TODO: split structure from the potential model (in design)
  """
  def __init__(self, filename: Path) -> None:

    # Initialization
    self.filename = Path(filename)
    self._settings = None    # A dictionary representation of the NNP settgins including descriptor, scaler, and model
    self.descriptor = None   # A dictionary of {element: Descriptor}   # TODO: short and long descriptors
    self.scaler = None       # A dictionary of {element: Scaler}       # TODO: short and long scalers
    self.model = None        # A dictionary of {element: Model}        # TODO: short and long models
    self.trainer = None

    logger.info(f"Initializing {self.__class__.__name__}(filename={self.filename})")
    # TODO: set formats as class variable or **kwargs
    self.scaler_save_format = "scaling.{:03d}.data"
    self.model_save_format = "weights.{:03d}.zip"

    if self._settings is None:
      self._read_settings_file()
      self._construct_descriptor()
      self._construct_scaler()
      self._construct_model()
      self._construct_trainer()

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
    # TODO: center & scaler
    _to_scaler_type = {  
      'center_symmetry_functions': 'center',
      'scale_symmetry_functions': 'scale',
      'scale_center_symmetry_functions': 'scale_center',
      'scale_center_symmetry_functions_sigma': 'scale_center_sigma',
    }
    # Defaults
    self._settings = defaultdict(list)
    self._settings.update({
      "epochs": 1,
      "updater_type": 0,
      "gradient_type": 1, 
    })
 
    # Read setting file
    logger.info(f"Reading NNP settings file:'{self.filename}'")
    with open(str(self.filename), 'r') as file:

      while True:
        # Read the next line
        line = file.readline()
        if not line:
          break
        
        # Read descriptor parameters
        # TODO: improve how to handle keyword and values (apply DRY)
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
        elif keyword == "center_symmetry_functions":
          self._settings["scale_type"] = _to_scaler_type[keyword]
        elif keyword == "scale_symmetry_functions":
          self._settings["scale_type"] = _to_scaler_type[keyword]
        elif keyword == "scale_center_symmetry_functions":
          self._settings["scale_type"] = _to_scaler_type[keyword]
        elif keyword == "scale_center_symmetry_functions_sigma":
          self._settings["scale_type"] = _to_scaler_type[keyword]
        elif keyword == "scale_min_short":
          self._settings[keyword] = float(tokens[0])
        elif keyword == "scale_max_short":
          self._settings[keyword] = float(tokens[0])
      
        # Read trainer
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
    logger.info(f"Registering symmetry functions (radial and angular)") # TODO: move logging inside .add() method
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
    logger.info(f"Creating descriptor scalers")
    for element in self._settings["elements"]:
      self.scaler[element] = DescriptorScaler(**kwargs) 

  def _construct_model(self) -> None:
    """
    Construct a neural network for each element using a dictionary representation of NNP settings.
    """
    self.model = {}
    # Instantiate neural network model for each element 
    logger.info(f"Creating neural network models")
    for element in self._settings["elements"]:
      input_size = self.descriptor[element].n_descriptor
      self.model[element] = NeuralNetworkModel(input_size, hidden_layers=((3, 't'), (3, 't')), output_layer=(1, 'l'))
      self.model[element].to(CFG["device"])
      # TODO: add element argument
      # TODO: read layers from the settings

  def _construct_trainer(self) -> None:
    """This method initializes a trainer instance including optimizer, loss function, criterion, etc.
    The created trainer instance when is used when fitting the energy models. 
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
        msg = "Gradient descent type fixed step is not implemented yet"
        logger.error(msg)
        raise NotImplementedError(msg)
    # Create trainer instance
    logger.debug(f"Preparing trainer kwargs={kwargs}")
    logger.info(f"Creating NNP trainer")
    self.trainer = NeuralNetworkPotentialTrainer(self, **kwargs)

  @Profiler.profile
  def fit_scaler(self, sloader: StructureLoader, save_scaler: bool = True,  **kwargs) -> None:
    """
    Fit scalers of descriptor for each element using the provided input structure loader.
    # TODO: split scaler, define it as separate step in pipeline
    """
    batch_size = kwargs.get("batch_size", 10)
    logger.info("Fitting descriptor scalers")
    for index, data in enumerate(sloader.get_data(), start=1):
      structure = Structure(
          data, 
          r_cutoff=self.r_cutoff,  # global cutoff radius (maximum) 
          requires_grad=False)     # there is NO NEED to keep the graph history (gradients) 
      for element in structure.elements:
        aids = structure.select(element).detach()
        for batch in create_batch(aids, batch_size):
          logger.debug(f"Structure={index}, element='{element}', batch={batch}")
          x = self.descriptor[element](structure, batch)
          self.scaler[element].fit(x)
    
    # Save scaler data into file  
    if save_scaler:
      self.save_scaler()

  def save_scaler(self):
    """
    Save scaler parameters for each element. 
    """
    # Save scaler parameters for each element separately
    for element in self.elements:
      logger.info(f"Saving scaler parameters for element: {element}")
      atomic_number = ElementMap.get_atomic_number(element)
      scaler_fn = Path(self.filename.parent, self.scaler_save_format.format(atomic_number)) 
      self.scaler[element].save(scaler_fn)
    # # Save scaler parameters for all element into a single file
    # scaler_fn = Path(self.filename.parent, self.scaler_save_format) 
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
    load scaler parameters for each element.
    No need to fit the scalers in this case. 
    """
    # Load scaler parameters for each element separately
    for element in self.elements:
      logger.info(f"Loading scaler parameters for element: {element}")
      atomic_number = ElementMap.get_atomic_number(element)
      scaler_fn = Path(self.filename.parent, self.scaler_save_format.format(atomic_number)) 
      self.scaler[element].load(scaler_fn)
    # # Load scaler parameters for all element into a single file
    # scaler_fn = Path(self.filename.parent, self.scaler_save_format) 
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
  def fit_model(self, sloader: StructureLoader, save_best_model: bool = True) -> Dict:
    """
    Fit the energy model for all elements using the provided structure loader.
    # TODO: avoid reading and calculating descriptor multiple times
    # TODO: descriptor element should be the same atom type as the aid
    # TODO: define a dataloader specific to energy and force data (shuffle, train & test split)
    # TODO: add validation output (MSE separate for force and energy)
    """
    history = self.trainer.fit(sloader, epochs=100)

    # TODO: save the best model by default, move to trainer
    if save_best_model:
      self.save_model()

    return history

  def save_model(self):
    """
    Save model weights separately for all element.
    """
    for element in self.elements:
      logger.info(f"Saving model weights for element: {element}")   
      atomic_number = ElementMap.get_atomic_number(element)
      model_fn = Path(self.filename.parent, self.model_save_format.format(atomic_number))
      self.model[element].save(model_fn)

  def load_model(self):
    """
    Load model weights separately for all element.
    """
    for element in self.elements:
        logger.info(f"Loading model weights for element: {element}")
        atomic_number = ElementMap.get_atomic_number(element)
        model_fn = Path(self.filename.parent, self.model_save_format.format(atomic_number))
        self.model[element].load(model_fn)

  def fit(self)  -> None:
    """
    Fit descriptor and model (if needed). 
    """
    pass

  def __call__(self, structure: Structure) -> Tensor:
    """
    Return the total energy of the given input structure.
    """
    # Loop over elements
    energy = None
    for element in self.elements:
      aids = structure.select(element).detach()
      x = self.descriptor[element](structure, aid=aids)
      x = self.scaler[element](x)
      x = self.model[element](x.float())
      x = torch.sum(x, dim=0)
      # TODO: float type neural network
      energy = x if energy is None else energy + x
    return energy

  @property
  def elements(self) -> List[str]:
    return self._settings['elements']

  @property
  def r_cutoff(self) -> float:
    """
    Return the maximum cutoff radius of all elemental descriptors.
    """
    return max([dsc.r_cutoff for dsc in self.descriptor.values()])


    

    




