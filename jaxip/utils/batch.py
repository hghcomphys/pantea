import math
from typing import Generator

from jaxip.types import Array


def create_batch(
    array: Array,
    batch_size: int,
) -> Generator[Array, None, None]:
    """
    Create baches of the input array.

    :param array: input array
    :type array: Array
    :param batch_size: desired batch size
    :type batch_size: int
    :yield: a batch of input array
    :rtype: Generator[Array, None, None]
    """
    n_batches = int(math.ceil(len(array) / batch_size))  # type: ignore
    for i in range(n_batches):
        yield array[i * batch_size : (i + 1) * batch_size, ...]
