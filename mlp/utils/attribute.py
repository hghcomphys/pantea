from ..logger import logger
from ..config import CFG
from typing import Dict
import torch


def set_tensors_as_attr(obj, tensors: Dict[str, torch.Tensor]) -> None:
  """
  An utility function to set an input dictionary of tensors as the class attributes.
  """
  logger.debug(f"Setting {len(tensors)} tensors as '{obj.__class__.__name__}'"
              f" class attributes: {', '.join(tensors.keys())}")
  for name, tensor in tensors.items():
    setattr(obj, name, tensor)


def cast_to_tensor(x):
  """
  An utility function to cast input variable (scaler, array, etc) 
  to torch tensor with predefined data and device types.
  """
  return torch.tensor(x, dtype=CFG['dtype'], device=CFG['device']) 