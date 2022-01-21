from .logger import logger
from typing import Any
import torch


def _get_device() -> torch.device:
  """
  Return and log available CUDA device.
  TODO: multiple GPU
  """
  is_cuda = torch.cuda.is_available()
  device = 'cuda' if is_cuda else 'cpu'
  logger.info(f"CUDA availability: {is_cuda}")
  logger.info(f"Default device: {device}")
  return torch.device(device)


class CFG:
  """
  A global configuration class of default values for variables.
  """
  __conf = {
    "device" : _get_device(),
    "dtype": torch.double,
  }
  __setters = list(__conf.keys()) # TODO: limit the setters?
 
  @staticmethod
  def get(name) -> Any:
    return CFG.__conf[name]

  @staticmethod
  def set(name, value) -> None:
    if name in CFG.__setters:
      logger.info(f"Setting default {name} as '{value}'")
      CFG.__conf[name] = value
    else:
      msg = f"Name '{name}' not accepted in the global configuration"
      logger.error(msg)
      raise NameError(msg)

  def __class_getitem__(cls, name) -> Any:
     return cls.get(name)

