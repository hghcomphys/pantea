from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial, update_wrapper
from typing import Callable, Mapping

import jax
import jax.numpy as jnp

from jaxip.pytree import BaseJaxPytreeDataClass, register_jax_pytree_node
from jaxip.types import Array

_TANH_PRE: float = ((math.e + 1 / math.e) / (math.e - 1 / math.e)) ** 3


def _hard(r: Array, r_cutoff: float) -> Array:
    return jnp.ones_like(r)


def _tanhu(r: Array, r_cutoff: float) -> Array:
    return jnp.tanh(1.0 - r / r_cutoff) ** 3


def _tanh(r: Array, r_cutoff: float) -> Array:
    return _TANH_PRE * jnp.tanh(1.0 - r / r_cutoff) ** 3


def _cos(r: Array, r_cutoff: float) -> Array:
    return 0.5 * (jnp.cos(jnp.pi * r / r_cutoff) + 1.0)


def _exp(r: Array, r_cutoff: float) -> Array:
    return jnp.exp(1.0 - 1.0 / (1.0 - (r / r_cutoff) ** 2))


def _poly1(r: Array, r_cutoff: float) -> Array:
    return (2.0 * r - 3.0) * r**2 + 1.0


def _poly2(r: Array, r_cutoff: float) -> Array:
    return ((15.0 - 6.0 * r) * r - 10) * r**3 + 1.0


_cutoff_kernel_map: Mapping[str, Callable] = {
    "hard": _hard,
    "tanhu": _tanhu,
    "tanh": _tanh,
    "cos": _cos,
    "exp": _exp,
    "poly1": _poly1,
    "poly2": _poly2,
}


def _wrapped_partial(kernel: Callable, r_cutoff: float) -> Callable:
    partial_kernel = partial(kernel, r_cutoff=r_cutoff)
    update_wrapper(partial_kernel, kernel)
    return partial_kernel


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
    kernel: Callable

    def __post_init__(self) -> None:
        self._assert_jit_dynamic_attributes()
        self._assert_jit_static_attributes(expected=("r_cutoff", "kernel"))

    @classmethod
    def from_cutoff_type(
        cls,
        r_cutoff: float,
        cutoff_type: str = "tanh",
    ) -> CutoffFunction:
        kernel = _wrapped_partial(
            _cutoff_kernel_map[cutoff_type],
            r_cutoff=r_cutoff,
        )
        return cls(r_cutoff, kernel)

    @jax.jit
    def __call__(self, r: Array) -> Array:
        return jnp.where(
            r < self.r_cutoff,
            self.kernel(r),
            jnp.zeros_like(r),
        )

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    @property
    def cutoff_type(self) -> str:
        kernel_name = self.kernel.__name__
        return kernel_name[1:] if kernel_name.startswith("_") else kernel_name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r_cutoff={self.r_cutoff})"


register_jax_pytree_node(CutoffFunction)
