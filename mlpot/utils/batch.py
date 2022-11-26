import math
import jax.numpy as jnp


def create_batch(array: jnp.ndarray, batch_size: int) -> jnp.ndarray:
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
