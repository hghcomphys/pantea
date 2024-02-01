from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp

from pantea.descriptors.acsf.cutoff import CutoffFunction
from pantea.pytree import BaseJaxPytreeDataclass, register_jax_pytree_node
from pantea.types import Array


class AngularSymmetryFunctionInterface(Protocol):
    """An expected interface for `three body` (angular) symmetry functions."""

    r_cutoff: Array

    def __call__(self, rij: Array, rik: Array, rjk: Array, cost: Array) -> Array:
        ...


@dataclass
class G3(BaseJaxPytreeDataclass, AngularSymmetryFunctionInterface):
    """Angular symmetry function."""

    eta: Array
    zeta: Array
    lambda0: Array
    r_shift: Array
    cutoff_function: CutoffFunction

    def __post_init__(self) -> None:
        self._assert_jit_dynamic_attributes(("eta", "zeta", "lambda0", "r_shift"))
        self._assert_jit_static_attributes(("cutoff_function",))
        self._cast_dynamic_attributes_to_array(self)

    @jax.jit
    def __call__(self, rij: Array, rik: Array, rjk: Array, cost: Array) -> Array:
        return (
            2.0 ** (1.0 - self.zeta)
            * jnp.power(1 + self.lambda0 * cost, self.zeta)
            * jnp.exp(-self.eta * (rij**2 + rik**2 + rjk**2))
            * self.cutoff_function(rij)
            * self.cutoff_function(rik)
            * self.cutoff_function(rjk)
        )

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    @property
    def r_cutoff(self) -> Array:
        return self.cutoff_function.r_cutoff


@dataclass
class G9(BaseJaxPytreeDataclass, AngularSymmetryFunctionInterface):
    """
    Modified angular symmetry function.

    See `J. Behler, J. Chem. Phys. 134, 074106 (2011)`.
    """

    eta: Array
    zeta: Array
    lambda0: Array
    r_shift: Array
    cutoff_function: CutoffFunction

    def __post_init__(self) -> None:
        self._assert_jit_dynamic_attributes(("eta", "zeta", "lambda0", "r_shift"))
        self._assert_jit_static_attributes(("cutoff_function",))
        self._cast_dynamic_attributes_to_array(self)

    @jax.jit
    def __call__(self, rij: Array, rik: Array, rjk: Array, cost: Array) -> Array:
        # TODO: r_shift, define params argument instead
        return (
            2.0 ** (1.0 - self.zeta)
            * jnp.power(1 + self.lambda0 * cost, self.zeta)
            * jnp.exp(-self.eta * (rij**2 + rik**2))
            * self.cutoff_function(rij)
            * self.cutoff_function(rik)
        )

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    @property
    def r_cutoff(self) -> Array:
        return self.cutoff_function.r_cutoff


register_jax_pytree_node(G3)
register_jax_pytree_node(G9)
