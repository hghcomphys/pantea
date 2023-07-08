import os
from functools import partial

import jax
import jax.numpy as jnp

from jaxip.atoms._structure import _calculate_distances
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


# @partial(jax.jit, static_argnums=0)
def _calculate_distances_and_masks(
    obj, structure: Structure
) -> tuple[Array, Array, Array]:
    r, R = _calculate_distances(
        atom_positions=structure.positions,
        neighbor_positions=structure.positions,
        lattice=structure.lattice,
    )
    masks = (0 < r) & (r < obj.r_cutoff)
    return r, R, masks


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
        r, _, masks = _calculate_distances_and_masks(self, structure)
        pair_energies = _compute_pair_energies(self, r)
        return 0.5 * jnp.where(masks, pair_energies, 0.0).sum()  # type: ignore

    @partial(jax.jit, static_argnums=0)
    def compute_forces(self, structure: Structure) -> Array:
        """Compute force component for each atoms."""
        r, R, masks = _calculate_distances_and_masks(self, structure)
        pair_forces = jnp.where(
            jnp.expand_dims(masks, axis=-1),
            _compute_pair_forces(self, r, R),
            jnp.zeros_like(R),
        )
        return jnp.sum(pair_forces, axis=1)  # type: ignore
