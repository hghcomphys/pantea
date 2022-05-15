import math
import torch
from torch import Tensor


def create_batch(array: Tensor , batch_size: int) -> Tensor:
  """
  Return the input array in form of batches (generator)
  """
  nb = int(math.ceil(len(array)/batch_size))
  for i in range(nb):
    yield array[i*batch_size:(i+1)*batch_size, ...]