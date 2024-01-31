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

    cfn: CutoffFunction
    eta: float
    zeta: float
    lambda0: float
    r_shift: float

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    @jax.jit
    def __call__(self, rij: Array, rik: Array, rjk: Array, cost: Array) -> Array:
        return (
            2.0 ** (1.0 - self.zeta)
            * jnp.power(1 + self.lambda0 * cost, self.zeta)
            * jnp.exp(-self.eta * (rij**2 + rik**2 + rjk**2))
            * self.cfn(rij)
            * self.cfn(rik)
            * self.cfn(rjk)
        )

    @property
    def r_cutoff(self) -> Array:
        return self.cfn.r_cutoff


@dataclass
class G9(BaseJaxPytreeDataclass, AngularSymmetryFunctionInterface):
    """
    Modified angular symmetry function.

    See `J. Behler, J. Chem. Phys. 134, 074106 (2011)`.
    """

    cfn: CutoffFunction
    eta: float
    zeta: float
    lambda0: float
    r_shift: float

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    @jax.jit
    def __call__(self, rij: Array, rik: Array, rjk: Array, cost: Array) -> Array:
        # TODO: r_shift, define params argument instead
        return (
            2.0 ** (1.0 - self.zeta)
            * jnp.power(1 + self.lambda0 * cost, self.zeta)
            * jnp.exp(-self.eta * (rij**2 + rik**2))
            * self.cfn(rij)
            * self.cfn(rik)
        )

    @property
    def r_cutoff(self) -> Array:
        return self.cfn.r_cutoff


register_jax_pytree_node(G3)
register_jax_pytree_node(G9)
