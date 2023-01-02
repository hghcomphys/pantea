from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from mlpot.base import register_jax_pytree_node
from mlpot.descriptors.acsf.cutoff import CutoffFunction
from mlpot.descriptors.acsf.symmetry import SymmetryFunction
from mlpot.types import Array


class AngularSymmetryFunction(SymmetryFunction, metaclass=ABCMeta):
    """A base class for `three body` (angular) symmetry functions."""

    # TODO: add other variant of angular symmetry functions (see N2P2 documentation).

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    @abstractmethod
    def __call__(
        self,
        rij: Array,
        rik: Array,
        rjk: Array,
        cost: Array,
    ) -> Array:
        pass


@dataclass
class G3(AngularSymmetryFunction):
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
    def __call__(
        self,
        rij: Array,
        rik: Array,
        rjk: Array,
        cost: Array,
    ) -> Array:
        return (
            2.0 ** (1.0 - self.zeta)
            * jnp.power(1 + self.lambda0 * cost, self.zeta)
            * jnp.exp(-self.eta * (rij**2 + rik**2 + rjk**2))
            * self.cfn(rij)
            * self.cfn(rik)
            * self.cfn(rjk)
        )


@dataclass
class G9(G3):
    """
    Modified angular symmetry function.

    J. Behler, J. Chem. Phys. 134, 074106 (2011).
    """

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    @jax.jit
    def __call__(
        self,
        rij: Array,
        rik: Array,
        rjk: Array,
        cost: Array,
    ) -> Array:
        # TODO: r_shift, define params argument instead
        return (
            2.0 ** (1.0 - self.zeta)
            * jnp.power(1 + self.lambda0 * cost, self.zeta)
            * jnp.exp(-self.eta * (rij**2 + rik**2))
            * self.cfn(rij)
            * self.cfn(rik)
        )


register_jax_pytree_node(G3)
register_jax_pytree_node(G9)
