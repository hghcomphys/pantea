import os
from functools import partial

import jax
import jax.numpy as jnp

from jaxip.atoms.structure import Structure
from jaxip.types import Array
from jaxip.units import units

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"


@partial(jax.jit, static_argnums=0)
def _compute_pair_energy(obj, r: Array) -> Array:
    term = obj.sigma / r
    term6 = term**6
    return 4.0 * obj.epsilon * term6 * (term6 - 1.0)


@partial(jax.jit, static_argnums=0)
def _compute_pair_force(obj, r: Array, R: Array) -> Array:
    term = obj.sigma / r
    term6 = term**6
    force_factor = -24.0 * obj.epsilon / (r * r) * term6 * (2.0 * term6 - 1.0)
    return jnp.expand_dims(force_factor, axis=-1) * R


class LJPotential:
    def __init__(
        self,
        sigma: float,
        epsilon: float,
        r_cutoff: float,
    ) -> None:
        self.sigma = sigma
        self.epsilon = epsilon
        self.r_cutoff = r_cutoff

    def __call__(self, structure: Structure) -> Array:
        r, _ = structure.calculate_distance(atom_index=jnp.arange(structure.natoms))
        mask = (0 < r) & (r < self.r_cutoff)
        pair_energies = _compute_pair_energy(self, r)
        return 0.5 * jnp.where(mask, pair_energies, 0.0).sum()  # type: ignore

    def compute_force(self, structure: Structure) -> Array:
        r, R = structure.calculate_distance(atom_index=jnp.arange(structure.natoms))
        mask = (0 < r) & (r < self.r_cutoff)
        pair_forces = jnp.where(
            jnp.expand_dims(mask, axis=-1),
            _compute_pair_force(self, r, R),
            jnp.zeros_like(R),
        )
        return jnp.sum(pair_forces, axis=1)  # type: ignore
