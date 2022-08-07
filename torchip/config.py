from .logger import logger
from typing import Any
import torch
import numpy
    

class _CFG:
  """
  A base configuration class of default values for global variables.
  """
  # TODO: circular import error between CFG & logger
  
  _conf = {
    # Global variables!
  }

  def __init__(self):
    self._set_attributes()

  def _set_attributes(self):
    for atr in self._conf:
      setattr(self, atr, self._conf[atr])
  
  def get(self, name) -> Any:
    return self._conf[name]

  def set(self, name, value) -> None:
    if name in self._conf.keys():
      logger.debug(f"Setting {name}: '{value}'")
      self._conf[name] = value
    else:
      logger.error(f"Name '{name}' is not allowed!", exception=NameError)

  def __getitem__(self, name) -> Any:
     return self.get(name)


class DataType(_CFG):
  """
  A configuration class for the tensors' data type.
  """
  _conf = {
    "FLOAT" : torch.float,
    "INT"    : torch.long,
    "UINT"   : torch.long,
    "INDEX"  : torch.long,
  }
dtype = DataType()  # create global dtype config


class Device(_CFG):
  """
  A configuration class for the tensors' device.
  TODO: further adjustments regarding mutiple GPUs are possible from here.
  """
  _conf = {
    "CPU" : torch.device('cpu'),
    "GPU" : torch.device('cuda'),
    "DEVICE" : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
  }
device = Device()  # create global device config


def manual_seed(seed: int) -> None:
  """
  Set the seed for generating random numbers.

  :param seed: random seed
  :type seed: int
  """  
  logger.debug("Setting the global random seed to {seed}")
  numpy.random.seed(seed)
  torch.manual_seed(seed)


class TaskClient:
  client =  None
  # from dask.distributed import Client
  # client = Client(memory_limit='4GB', n_workers=2, processes=True, threads_per_worker=2, dashboard_address=':8791')
  # FIXME: There is an issue in Dask where the pytorch graph history cannot be transferred to 
  # the client workers when having multiple processes





