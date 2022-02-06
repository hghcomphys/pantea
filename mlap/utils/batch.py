import numpy as np

def create_batch(array: np.ndarray, batch_size: int) -> np.ndarray:
  """
  Return the input array in form of batches (generator)
  """
  nb = int(np.ceil(len(array)/batch_size))
  for i in range(nb):
    yield array[i*batch_size:(i+1)*batch_size, ...]