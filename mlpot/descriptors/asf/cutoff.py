import math
import jax
import jax.numpy as jnp
from mlpot.logger import logger
from mlpot.base import _Base
from functools import partial


class CutoffFunction(_Base):
    """
    This class contains different cutoff functions used for ASF descriptor.
    See N2P2 -> https://compphysvienna.github.io/n2p2/api/cutoff_functions.html?highlight=cutoff#
                https://compphysvienna.github.io/n2p2/topics/keywords.html?highlight=cutoff_type
    """

    # TODO: add logger
    # TODO: add poly 3 & 4 funcions

    _TANH_PRE = ((math.e + 1 / math.e) / (math.e - 1 / math.e)) ** 3

    def __init__(self, r_cutoff: float, cutoff_type: str = "tanh"):
        self.r_cutoff = r_cutoff
        self.cutoff_type = cutoff_type.lower()
        # Set cutoff type function
        try:
            self.cutoff_function = getattr(self, f"{self.cutoff_type}")
        except AttributeError:
            logger.error(
                f"'{self.__class__.__name__}' has no cutoff function '{self.cutoff_type}'",
                exception=NotImplementedError,
            )

    @partial(jax.jit, static_argnums=(0,))  # FIXME
    def __call__(self, r: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(r < self.r_cutoff, self.cutoff_function(r), jnp.zeros_like(r))

    def hard(self, r: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones_like(r)

    def tanhu(self, r: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(1.0 - r / self.r_cutoff) ** 3

    def tanh(self, r: jnp.ndarray) -> jnp.ndarray:
        return self._TANH_PRE * jnp.tanh(1.0 - r / self.r_cutoff) ** 3

    def cos(self, r: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * (jnp.cos(jnp.pi * r / self.r_cutoff) + 1.0)

    def exp(self, r: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(1.0 - 1.0 / (1.0 - (r / self.r_cutoff) ** 2))

    def poly1(self, r: jnp.ndarray) -> jnp.ndarray:
        return (2.0 * r - 3.0) * r**2 + 1.0

    def poly2(self, r: jnp.ndarray) -> jnp.ndarray:
        return ((15.0 - 6.0 * r) * r - 10) * r**3 + 1.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r_cutoff={self.r_cutoff}, cutoff_type='{self.cutoff_type}')"
