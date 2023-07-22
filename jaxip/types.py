from dataclasses import dataclass

import jax.numpy as jnp
from jax._src.numpy.lax_numpy import _ScalarMeta
from jaxlib.xla_extension import ArrayImpl

Array = ArrayImpl
Dtype = _ScalarMeta
Element = str
Scalar = ArrayImpl


@dataclass
class DataType:
    """
    A configuration for array data type.

    It is globally used as default dtype for arrays, indices, etc.
    User can modify any default dtype, for example, via setting the global
    floating point precision (`FLOATX`) to a single (float32)
    or double (float64).
    """

    FLOATX: Dtype = jnp.float32
    INT: Dtype = jnp.int32
    UINT: Dtype = jnp.uint32
    INDEX: Dtype = jnp.int32


_dtype = DataType()
