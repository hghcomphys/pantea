import math
from torch import Tensor


def create_batch(array: Tensor , batch_size: int) -> Tensor:
  """
  Return the input array in form of batches (generator).

  Args:
      array (Tensor): input array
      batch_size (int): desired batch size

  Yields:
      Iterator[Tensor]: a batch of input array
  """  
  nb = int(math.ceil(len(array)/batch_size))
  for i in range(nb):
    yield array[i*batch_size:(i+1)*batch_size, ...]