from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxip.base import register_jax_pytree_node
from jaxip.descriptors.acsf.cutoff import CutoffFunction
from jaxip.descriptors.acsf.symmetry import SymmetryFunction
from jaxip.types import Array


class RadialSymmetryFunction(SymmetryFunction, metaclass=ABCMeta):
    """A base class for `two body` (radial) symmetry functions."""

    # TODO: define generic **params input arguments in the base class?
    # TODO: define a internal cutoff radius
    # TODO: add other variant of radial symmetry functions.
    # TODO: add logging when initializing each symmetry function.
    @abstractmethod
    def __call__(self, rij: Array) -> Array:
        pass


@dataclass
class G1(RadialSymmetryFunction):
    """Plain cutoff function as symmetry function."""

    cfn: CutoffFunction

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    @jax.jit
    def __call__(self, rij: Array) -> Array:
        return self.cfn(rij)


@dataclass
class G2(RadialSymmetryFunction):
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


register_jax_pytree_node(G1)
register_jax_pytree_node(G2)
