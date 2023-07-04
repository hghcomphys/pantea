from typing import Any, Dict, Optional

import jax.numpy as jnp

from jaxip.logger import logger
from jaxip.types import Array, Dtype
from jaxip.types import _dtype


def set_as_attribute(
    obj: Any,
    items: Dict[str, Any],
    prefix: str = "",
    postfix: str = "",
) -> None:
    """
    An utility function to set a dictionary of items as the input object attributes.


    :param obj: instance
    :param items: dictionary of attributes
    :type items: Dict[str, Any]
    :param prefix: _description_, defaults to ""
    :type prefix: str, optional
    :param postfix: _description_, defaults to ""
    :type postfix: str, optional
    """
    logger.debug(f"Setting {len(items)} items as {obj.__class__.__name__} attributes:")
    for name, item in items.items():
        attr_name: str = f"{prefix}{name}{postfix}"
        logger.debug(f"-> {obj.__class__.__name__}.{attr_name}")
        setattr(obj, attr_name, item)


def asarray(data: Any, dtype: Optional[Dtype] = None) -> Array:
    """
    An utility function to cast input data (scalar, array, etc) to Array type with predefined dtype.

    :param value: input data
    :type value: Any
    :param dtype: casted dtype. Default dtype will be used otherwise.
    :type dtype: Optional[Dtype], optional
    :return: casted input
    :rtype: Array
    """
    if dtype is None:
        dtype = _dtype.FLOATX
    return jnp.asarray(data, dtype=dtype)
