from typing import Mapping

import jax.numpy as jnp

from mlpot.config import _CFG

Array = jnp.ndarray
Dtype = jnp.dtype


class DataType(_CFG):
    """
    A configuration class for the tensors' data type.
    """

    _conf: Mapping[str, jnp.dtype] = {
        "FLOATX": jnp.float32,
        "INT": jnp.int32,
        "UINT": jnp.uint32,
        "INDEX": jnp.int32,
    }


# Create global dtype config
dtype: DataType = DataType()
