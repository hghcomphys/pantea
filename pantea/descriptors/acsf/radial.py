from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp

from pantea.descriptors.acsf.cutoff import CutoffFunction
from pantea.pytree import BaseJaxPytreeDataclass, register_jax_pytree_node
from pantea.types import Array


class RadialSymmetryFunctionInterface(Protocol):
    """An expected interface for `two body` (radial) symmetry functions."""

    r_cutoff: Array

    def __call__(self, rij: Array) -> Array:
        ...


@dataclass
class G1(BaseJaxPytreeDataclass, RadialSymmetryFunctionInterface):
    """Plain cutoff function as symmetry function."""

    cutoff_function: CutoffFunction

    def __post_init__(self) -> None:
        self._assert_jit_dynamic_attributes()
        self._assert_jit_static_attributes(("cutoff_function",))

    @jax.jit
    def __call__(self, rij: Array) -> Array:
        return self.cutoff_function(rij)

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    @property
    def r_cutoff(self) -> Array:
        return self.cutoff_function.r_cutoff


@dataclass
class G2(BaseJaxPytreeDataclass, RadialSymmetryFunctionInterface):
    """Radial exponential symmetry function."""

    eta: Array
    r_shift: Array
    cutoff_function: CutoffFunction

    def __post_init__(self) -> None:
        self._assert_jit_dynamic_attributes(("eta", "r_shift"))
        self._assert_jit_static_attributes(("cutoff_function",))
        self._cast_dynamic_attributes_to_array(self)

    @jax.jit
    def __call__(self, rij: Array) -> Array:
        cfn = self.cutoff_function(rij)
        return jnp.exp(-self.eta * (rij - self.r_shift) ** 2) * cfn

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    @property
    def r_cutoff(self) -> Array:
        return self.cutoff_function.r_cutoff


register_jax_pytree_node(G1)
register_jax_pytree_node(G2)
