import os
from typing import NamedTuple

import jax
import jax.numpy as jnp

from pantea.atoms.neighbor import _calculate_masks_with_aux_from_structure
from pantea.atoms.structure import Structure
from pantea.types import Array

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"


class LJPotentialParams(NamedTuple):
    epsilon: Array
    sigma: Array


# @jax.jit
def _compute_pair_energies(params: LJPotentialParams, r: Array) -> Array:
    term = params.sigma / r
    term6 = term**6
    return 4.0 * params.epsilon * term6 * (term6 - 1.0)


@jax.jit
def _compute_total_energy(
    params: LJPotentialParams,
    positions: Array,
    lattice: Array,
    r_cutoff: Array,
) -> Array:
    masks, (rij, _) = _calculate_masks_with_aux_from_structure(
        positions, r_cutoff, lattice
    )
    pair_energies = _compute_pair_energies(params, rij)
    return 0.5 * jnp.sum(jnp.where(masks, pair_energies, 0.0))  # type: ignore


# @jax.jit
def _compute_pair_forces(params: LJPotentialParams, r: Array, R: Array) -> Array:
    term = params.sigma / r
    term6 = term**6
    force_factor = -24.0 * params.epsilon / (r * r) * term6 * (2.0 * term6 - 1.0)
    return jnp.expand_dims(force_factor, axis=-1) * R


@jax.jit
def _compute_forces(
    params: LJPotentialParams,
    positions: Array,
    lattice: Array,
    r_cutoff: Array,
) -> Array:
    masks, (rij, Rij) = _calculate_masks_with_aux_from_structure(
        positions, r_cutoff, lattice
    )
    pair_forces = jnp.where(
        jnp.expand_dims(masks, axis=-1),
        _compute_pair_forces(params, rij, Rij),
        jnp.zeros_like(Rij),
    )
    return jnp.sum(pair_forces, axis=1)  # type: ignore


class LJPotential:
    """An implementation of Lennard-Jones potential for a single type atom."""

    def __init__(
        self,
        sigma: Array,
        epsilon: Array,
        r_cutoff: Array,
    ) -> None:
        self.sigma = jnp.array(sigma)
        self.epsilon = jnp.array(epsilon)
        self.r_cutoff = jnp.array(r_cutoff)

    def __call__(self, structure: Structure) -> Array:
        """Compute total potential energy."""
        return _compute_total_energy(
            LJPotentialParams(self.epsilon, self.sigma),
            structure.positions,
            structure.lattice,
            self.r_cutoff,
        )

    def compute_forces(self, structure: Structure) -> Array:
        """Compute force component for each atoms."""
        return _compute_forces(
            LJPotentialParams(self.epsilon, self.sigma),
            structure.positions,
            structure.lattice,
            self.r_cutoff,
        )
