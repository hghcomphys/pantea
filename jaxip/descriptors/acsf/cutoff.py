import math
from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jaxip.base import _BaseJaxPytreeDataClass, register_jax_pytree_node
from jaxip.logger import logger
from jaxip.types import Array

_TANH_PRE: float = ((math.e + 1 / math.e) / (math.e - 1 / math.e)) ** 3


@dataclass
class CutoffFunction(_BaseJaxPytreeDataClass):
    """Cutoff function for ACSF descriptor.

    See `cutoff function`_ and `cutoff type`_ for more details.

    .. _`cutoff function`: https://compphysvienna.github.io/n2p2/api/cutoff_functions.html?highlight=cutoff#
    .. _`cutoff type`: https://compphysvienna.github.io/n2p2/topics/keywords.html?highlight=cutoff_type
    """

    # TODO: add logger
    # TODO: add poly 3 & 4 functions

    r_cutoff: float
    cutoff_type: str = "tanh"
    cutoff_function: Optional[Callable] = None  # lambda r: r

    def __post_init__(self) -> None:
        self.cutoff_type = self.cutoff_type.lower()
        if self.cutoff_function is None:
            try:
                self.cutoff_function = getattr(self, f"{self.cutoff_type}")
            except AttributeError:
                logger.error(
                    f"'{self.__class__.__name__}' has no cutoff function '{self.cutoff_type}'",
                    exception=NotImplementedError,
                )

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    @jax.jit
    def __call__(self, r: Array) -> Array:
        return jnp.where(
            r < self.r_cutoff,
            self.cutoff_function(r),  # type: ignore
            jnp.zeros_like(r),
        )

    def hard(self, r: Array) -> Array:
        return jnp.ones_like(r)

    def tanhu(self, r: Array) -> Array:
        return jnp.tanh(1.0 - r / self.r_cutoff) ** 3

    def tanh(self, r: Array) -> Array:
        return _TANH_PRE * jnp.tanh(1.0 - r / self.r_cutoff) ** 3

    def cos(self, r: Array) -> Array:
        return 0.5 * (jnp.cos(jnp.pi * r / self.r_cutoff) + 1.0)

    def exp(self, r: Array) -> Array:
        return jnp.exp(1.0 - 1.0 / (1.0 - (r / self.r_cutoff) ** 2))

    def poly1(self, r: Array) -> Array:
        return (2.0 * r - 3.0) * r**2 + 1.0

    def poly2(self, r: Array) -> Array:
        return ((15.0 - 6.0 * r) * r - 10) * r**3 + 1.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r_cutoff={self.r_cutoff}, cutoff_type='{self.cutoff_type}')"


register_jax_pytree_node(CutoffFunction)
