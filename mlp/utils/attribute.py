from ..logger import logger
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
