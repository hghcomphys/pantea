import math
from typing import Generator

from jaxip.types import Array


def create_batch(array: Array, batch_size: int) -> Generator[Array, None, None]:
    """
    Return the input array in form of batches (generator).

    Args:
        array (jnp.ndarray): input array
        batch_size (int): desired batch size

    Yields:
        Iterator[jnp.ndarray]: a batch of input array
    """
    nb = int(math.ceil(len(array) / batch_size))
    for i in range(nb):
        yield array[i * batch_size : (i + 1) * batch_size, ...]
