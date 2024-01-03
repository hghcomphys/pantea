import os
from functools import partial
from typing import Protocol

import jax
import jax.numpy as jnp

from pantea.atoms.neighbor import _calculate_masks_with_aux_from_structure
from pantea.atoms.structure import Structure
from pantea.types import Array

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"


class LJPotentialParams(Protocol):
    epsilon: float
    sigma: float


# @partial(jax.jit, static_argnums=0)
def _compute_pair_energies(params: LJPotentialParams, r: Array) -> Array:
    term = params.sigma / r
    term6 = term**6
    return 4.0 * params.epsilon * term6 * (term6 - 1.0)


# @partial(jax.jit, static_argnums=0)
def _compute_pair_forces(params: LJPotentialParams, r: Array, R: Array) -> Array:
    term = params.sigma / r
    term6 = term**6
    force_factor = -24.0 * params.epsilon / (r * r) * term6 * (2.0 * term6 - 1.0)
    return jnp.expand_dims(force_factor, axis=-1) * R


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
        masks, (rij, _) = _calculate_masks_with_aux_from_structure(
            structure.positions, self.r_cutoff, structure.lattice
        )
        pair_energies = _compute_pair_energies(self, rij)
        return 0.5 * jnp.where(masks, pair_energies, 0.0).sum()  # type: ignore

    @partial(jax.jit, static_argnums=0)
    def compute_forces(self, structure: Structure) -> Array:
        """Compute force component for each atoms."""
        masks, (rij, Rij) = _calculate_masks_with_aux_from_structure(
            structure.positions, self.r_cutoff, structure.lattice
        )
        pair_forces = jnp.where(
            jnp.expand_dims(masks, axis=-1),
            _compute_pair_forces(self, rij, Rij),
            jnp.zeros_like(Rij),
        )
        return jnp.sum(pair_forces, axis=1)  # type: ignore
