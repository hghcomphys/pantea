import os
from functools import partial
from typing import Tuple, cast

import jax
import jax.numpy as jnp

from jaxip.atoms.neighbor import Neighbor
from jaxip.atoms.structure import Structure
from jaxip.types import Array

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"


# @partial(jax.jit, static_argnums=0)
def _compute_pair_energies(obj, r: Array) -> Array:
    term = obj.sigma / r
    term6 = term**6
    return 4.0 * obj.epsilon * term6 * (term6 - 1.0)


# @partial(jax.jit, static_argnums=0)
def _compute_pair_forces(obj, r: Array, R: Array) -> Array:
    term = obj.sigma / r
    term6 = term**6
    force_factor = -24.0 * obj.epsilon / (r * r) * term6 * (2.0 * term6 - 1.0)
    return jnp.expand_dims(force_factor, axis=-1) * R


def _get_neighbor_data(
    structure: Structure,
    r_cutoff: float,
) -> Tuple[Array, Array, Array]:
    if structure.neighbor is None:
        structure.update_neighbor(r_cutoff)
    neighbor = cast(Neighbor, structure.neighbor)
    return (neighbor.masks, neighbor.rij, neighbor.Rij)


class LJPotential:
    """An implementation of Lennard-Jones potential."""

    def __init__(
        self,
        sigma: float,
        epsilon: float,
        r_cutoff: float,
    ) -> None:
        self.sigma = sigma
        self.epsilon = epsilon
        self.r_cutoff = r_cutoff

    @partial(jax.jit, static_argnums=0)
    def __call__(self, structure: Structure) -> Array:
        """Compute total energy."""
        masks, rij, _ = _get_neighbor_data(structure, self.r_cutoff)
        pair_energies = _compute_pair_energies(self, rij)
        return 0.5 * jnp.where(masks, pair_energies, 0.0).sum()  # type: ignore

    @partial(jax.jit, static_argnums=0)
    def compute_forces(self, structure: Structure) -> Array:
        """Compute force component for each atoms."""
        masks, rij, Rij = _get_neighbor_data(structure, self.r_cutoff)
        pair_forces = jnp.where(
            jnp.expand_dims(masks, axis=-1),
            _compute_pair_forces(self, rij, Rij),
            jnp.zeros_like(Rij),
        )
        return jnp.sum(pair_forces, axis=1)  # type: ignore
