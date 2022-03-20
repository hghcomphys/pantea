import math
import torch


def create_batch(array: torch.Tensor , batch_size: int) -> torch.Tensor:
  """
  Return the input array in form of batches (generator)
  """
  nb = int(math.ceil(len(array)/batch_size))
  for i in range(nb):
    yield array[i*batch_size:(i+1)*batch_size, ...]