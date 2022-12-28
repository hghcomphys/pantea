from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from mlpot.base import register_jax_pytree_node
from mlpot.descriptors.acsf.cutoff import CutoffFunction
from mlpot.descriptors.acsf.symmetry import SymmetryFunction
from mlpot.types import Array


class RadialElements(NamedTuple):
    central_i: str
    neighbor_j: str


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

    cutoff_function: CutoffFunction

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    @jax.jit
    def __call__(self, rij: Array) -> Array:
        return self.cutoff_function(rij)


@dataclass
class G2(RadialSymmetryFunction):
    """Radial exponential symmetry function."""

    cutoff_function: CutoffFunction
    r_shift: float
    eta: float

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    @jax.jit
    def __call__(self, rij: Array) -> Array:
        return jnp.exp(-self.eta * (rij - self.r_shift) ** 2) * self.cutoff_function(
            rij
        )


register_jax_pytree_node(G1)
register_jax_pytree_node(G2)
