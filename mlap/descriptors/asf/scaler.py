
from typing import Dict
from collections import defaultdict
import numpy as np
import torch


class AtomicSymmetryFunctionScaler:
  """
  Scale descriptor values.
  """
  def __init__(self, **kwargs):
    self.min = defaultdict(np.inf)
    self.max = defaultdict(-np.inf)
    self.mean = defaultdict(float)
    self.sigma = defaultdict(float)
  
  def fit(self, values: Dict[str, torch.Tensor]) -> None:
    pass

  def transform(self, values: Dict[str, torch.Tensor]):
    pass

