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

    cfn: CutoffFunction

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    @jax.jit
    def __call__(self, rij: Array) -> Array:
        return self.cfn(rij)

    @property
    def r_cutoff(self) -> Array:
        return self.cfn.r_cutoff


@dataclass
class G2(BaseJaxPytreeDataclass, RadialSymmetryFunctionInterface):
    """Radial exponential symmetry function."""

    cfn: CutoffFunction
    r_shift: float
    eta: float

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    @jax.jit
    def __call__(self, rij: Array) -> Array:
        return jnp.exp(-self.eta * (rij - self.r_shift) ** 2) * self.cfn(rij)

    @property
    def r_cutoff(self) -> Array:
        return self.cfn.r_cutoff


register_jax_pytree_node(G1)
register_jax_pytree_node(G2)
