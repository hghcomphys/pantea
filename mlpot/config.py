from .logger import logger
from typing import Any
import jax.numpy as jnp


class _CFG:
    """
    A configuration class of default values for global variables.
    """

    # TODO: circular import error between CFG & logger
    _conf = {
        # Global variables!
    }

    def __init__(self):
        self._set_attributes()

    def _set_attributes(self):
        for atr in self._conf:
            setattr(self, atr, self._conf[atr])

    def get(self, name) -> Any:
        return self._conf[name]

    def set(self, name, value) -> None:
        if name in self._conf.keys():
            logger.debug(f"Setting {name}: '{value}'")
            self._conf[name] = value
        else:
            logger.error(f"Name '{name}' is not allowed!", exception=NameError)

    def __getitem__(self, name) -> Any:
        return self.get(name)


class DataType(_CFG):
    """
    A configuration class for the tensors' data type.
    """

    _conf = {
        "FLOATX": jnp.float32,
        "INT": jnp.int32,
        "UINT": jnp.uint32,
        "INDEX": jnp.int32,
    }


# Create global dtype config
dtype = DataType()
