import math
from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from jaxip.logger import logger
from jaxip.pytree import BaseJaxPytreeDataClass, register_jax_pytree_node
from jaxip.types import Array

_TANH_PRE: float = ((math.e + 1 / math.e) / (math.e - 1 / math.e)) ** 3


@dataclass
class CutoffFunction(BaseJaxPytreeDataClass):
    """Cutoff function for ACSF descriptor.

    Cutoff functions are utilized in the calculation of Atom-centered Symmetry
    Function (ACSF) descriptors. These functions serve to limit the influence
    of atoms located beyond a specified distance from the central atom.

    The ACSF descriptors employ cutoff functions to determine the range within which
    neighboring atoms contribute to the descriptor calculation.
    The cutoff function assigns a weight to each neighbor atom based on its
    distance from the central atom. Typically, a smooth cutoff function is
    employed to smoothly taper off the contribution of
    atoms as they move away from the central atom.

    The choice of cutoff function can vary depending on the specific application.
    Examples of commonly used cutoff functions include the hyperbolic tangent (tanh)
    cutoff, exponential, or exponential.


    See `cutoff function`_ and `cutoff type`_ for more details.

    .. _`cutoff function`: https://compphysvienna.github.io/n2p2/api/cutoff_functions.html?highlight=cutoff#
    .. _`cutoff type`: https://compphysvienna.github.io/n2p2/topics/keywords.html?highlight=cutoff_type
    """

    r_cutoff: float
    cutoff_type: str = "tanh"
    cutoff_function: Optional[Callable] = None

    def __post_init__(self) -> None:
        if self.cutoff_function is None:
            try:
                self.cutoff_function = getattr(
                    self, f"{self.cutoff_type.lower()}"
                )
            except AttributeError:
                logger.error(
                    f"Unknown cutoff function '{self.cutoff_type}'",
                    exception=NotImplementedError,
                )
        else:
            self.cutoff_type = self.cutoff_function.__name__
        self._assert_jit_dynamic_attributes()
        self._assert_jit_static_attributes(
            expected=("r_cutoff", "cutoff_type", "cutoff_function")
        )

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

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(r_cutoff={self.r_cutoff}"
            f", cutoff_type='{self.cutoff_type}')"
        )


register_jax_pytree_node(CutoffFunction)
