from dataclasses import dataclass

import jax.numpy as jnp

Array = jnp.ndarray
Dtype = jnp.dtype
Element = str


@dataclass
class DataType:
    """
    A configuration class for arrays data type.
    It is globally used as default dtype for arrays, indices, etc.

    User can modify any default dtype, for example, via setting the global
    floating point precision of `FLOATX` to single (float32) or double (float64).
    """

    FLOATX: Dtype = jnp.float32  # type: ignore
    INT: Dtype = jnp.int32  # type: ignore
    UINT: Dtype = jnp.uint32  # type: ignore
    INDEX: Dtype = jnp.int32  # type: ignore


dtype: DataType = DataType()
