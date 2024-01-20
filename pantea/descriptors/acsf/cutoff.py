from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

import jax
import jax.numpy as jnp

from pantea.pytree import BaseJaxPytreeDataClass, register_jax_pytree_node
from pantea.types import Array

_TANH_PRE: Array = jnp.array(((math.e + 1 / math.e) / (math.e - 1 / math.e)) ** 3)


def _apply_cutoff(r: Array, fc: Array, r_cutoff: Array) -> Array:
    return jnp.where(r < r_cutoff, fc, jnp.zeros_like(r))


@jax.jit
def _hard(r: Array, r_cutoff: Array) -> Array:
    fc = jnp.ones_like(r)
    return _apply_cutoff(r, fc, r_cutoff)


@jax.jit
def _tanhu(r: Array, r_cutoff: Array) -> Array:
    fc = jnp.tanh(1.0 - r / r_cutoff) ** 3
    return _apply_cutoff(r, fc, r_cutoff)


@jax.jit
def _tanh(r: Array, r_cutoff: Array) -> Array:
    fc = _TANH_PRE * jnp.tanh(1.0 - r / r_cutoff) ** 3
    return _apply_cutoff(r, fc, r_cutoff)


@jax.jit
def _cos(r: Array, r_cutoff: Array) -> Array:
    fc = 0.5 * (jnp.cos(jnp.pi * r / r_cutoff) + 1.0)
    return _apply_cutoff(r, fc, r_cutoff)


@jax.jit
def _exp(r: Array, r_cutoff: Array) -> Array:
    fc = jnp.exp(1.0 - 1.0 / (1.0 - (r / r_cutoff) ** 2))
    return _apply_cutoff(r, fc, r_cutoff)


@jax.jit
def _poly1(r: Array, r_cutoff: Array) -> Array:
    fc = (2.0 * r - 3.0) * r**2 + 1.0
    return _apply_cutoff(r, fc, r_cutoff)


@jax.jit
def _poly2(r: Array, r_cutoff: Array) -> Array:
    fc = ((15.0 - 6.0 * r) * r - 10) * r**3 + 1.0
    return _apply_cutoff(r, fc, r_cutoff)


_MAP_CUTOFF_FUNCTIONS: Mapping[str, Callable[[Array, Array], Array]] = {
    "hard": _hard,
    "tanhu": _tanhu,
    "tanh": _tanh,
    "cos": _cos,
    "exp": _exp,
    "poly1": _poly1,
    "poly2": _poly2,
}


@dataclass
class CutoffFunction(BaseJaxPytreeDataClass):
    """Cutoff function for ACSF descriptor.

    Cutoff functions are utilized in the calculation of Atom-centered Symmetry
    Function (ACSF) descriptors. These functions serve to limit the influence
    of atoms located beyond a specified distance from the central atom.

    The ACSF descriptors employ cutoff functions to determine the range within which
    neighboring atoms contribute to the descriptor calculation. In fact,
    cutoff function assigns a weight to each neighbor atom based on its
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

    r_cutoff: Array
    cutoff_function: Callable

    @classmethod
    def from_type(
        cls,
        cutoff_type: str,
        r_cutoff: float,
    ) -> CutoffFunction:
        """Create cutoff function from the cutoff type such as "tanh", "cos", etc."""
        return cls(
            jnp.array(r_cutoff),
            _MAP_CUTOFF_FUNCTIONS[cutoff_type],
        )

    def __post_init__(self) -> None:
        self._assert_jit_dynamic_attributes(expected=("r_cutoff",))
        self._assert_jit_static_attributes(expected=("cutoff_function",))

    def __call__(self, r: Array) -> Array:
        return self.cutoff_function(r, self.r_cutoff)

    def __hash__(self) -> int:
        """Override the hash function from the base jax pytree data class."""
        return super().__hash__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r_cutoff={self.r_cutoff})"


register_jax_pytree_node(CutoffFunction)
