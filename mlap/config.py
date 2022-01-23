from .logger import logger
# from pathlib import Path
from typing import Any
import torch


class CFG:
  """
  A global configuration class of default values for variables.
  """
  __conf = {
    # Device
    "is_cuda": torch.cuda.is_available(),
    "device" : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    # Dtype
    "dtype": torch.double,
    "dtype_index": torch.long,
    # Logging TODO: circular import error between CFG & logger
    # "logging_level": logging.INFO,
    # "log_file": Path("mlap.log"),
  }
  __setters = list(__conf.keys()) # TODO: limit the setters?
 
  @staticmethod
  def get(name) -> Any:
    return CFG.__conf[name]

  @staticmethod
  def set(name, value) -> None:
    if name in CFG.__setters:
      logger.info(f"Resetting default {name} as '{value}'")
      CFG.__conf[name] = value
    else:
      msg = f"Name '{name}' not accepted in the global configuration"
      logger.error(msg)
      raise NameError(msg)

  def __class_getitem__(cls, name) -> Any:
     return cls.get(name)

