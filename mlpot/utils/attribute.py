from ..logger import logger
from ..config import dtype as _dtype
from typing import Dict, Any
import jax.numpy as jnp


def set_as_attribute(
    obj: Any,
    items: Dict[str, Any],
    prefix: str = "",
    postfix: str = "",
) -> None:
    """
    An utility function to set an input dictionary of items as the given object attributes.

    Args:
        obj (Any): an instance
        Any (Dict): a dictionary of items
    """
    logger.debug(f"Setting {len(items)} items as {obj.__class__.__name__} attributes:")
    for name, item in items.items():
        attr_name = f"{prefix}{name}{postfix}"
        logger.debug(f"-> {obj.__class__.__name__}.{attr_name}")
        setattr(obj, attr_name, item)


def cast_to_tensor(x, dtype: jnp.dtype = _dtype.FLOATX) -> jnp.ndarray:
    """
    An utility function to cast input variable (scalar, array, etc)
    to torch tensor with predefined data and device types.
    """
    return jnp.asarray(x, dtype=dtype)
