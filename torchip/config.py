from .logger import logger
from typing import Any
import torch


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
    "FLOATX" : torch.float,
    "INT"    : torch.long,
    "UINT"   : torch.long,
    "INDEX"  : torch.long,
  }


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
  

# from dask.distributed import Client
# class TaskClient:
#   client = Client(memory_limit='3GB', n_workers=4, processes=True, threads_per_worker=1, dashboard_address=':8791')


# Create dtype and device configurations
dtype = DataType()
device = Device()
