
from ..base import BaseTorchipClass
from torch import nn


class BaseModel (BaseTorchipClass, nn.Module):
  """
  A base class for all kinds of ML-based potential models.
  """
  pass