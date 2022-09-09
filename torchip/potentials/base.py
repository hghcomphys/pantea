from ..base import BaseTorchipClass
from collections import defaultdict
from typing import Dict

class Potential (BaseTorchipClass):
  """
  A base class that contains all required data and operations to train a ML-based potential 
  including structures, descriptors, models, etc. 
  All potentials must derive from this base class.
  """
  pass


class Settings (BaseTorchipClass):
  """
  A base class for potential settings.
  Each potential contains several parameters and settings that can be handled using the derived setting classes.
  """
  def __init__(self, default = Dict) -> None:
    self._settings = defaultdict(None)
    self._settings.update(default)

   