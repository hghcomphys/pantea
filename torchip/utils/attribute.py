from ..logger import logger
from ..config import dtype, device
from typing import Dict, Any
from torch import Tensor
import torch


def set_tensors_as_attr(obj: Any, tensors: Dict[str, Tensor]) -> None:
  """An utility function to set an input dictionary of tensors as the class attributes.

  Args:
      obj (Any): an instance
      tensors (Dict[str, Tensor]): a dictionary of tensors
  """  
  logger.debug(f"Setting {len(tensors)} tensors as '{obj.__class__.__name__}'"
              f" class attributes: {', '.join(tensors.keys())}")
  for name, tensor in tensors.items():
    setattr(obj, name, tensor)


def cast_to_tensor(x) -> Tensor:
  """An utility function to cast input variable (scaler, array, etc) 
  to torch tensor with predefined data and device types.

  Args:
      x (Any): input variable (e.g. scaler, array)

  Returns:
      Tensor: casted input to a tensor
  """  
  return torch.tensor(x, dtype=dtype.FLOATX, device=device.DEVICE) 