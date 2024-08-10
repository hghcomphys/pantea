import os
from typing import Callable, Literal, NamedTuple, Optional

import jax
import jax.numpy as jnp

from pantea.atoms.neighbor import _calculate_masks_with_aux_from_structure
from pantea.atoms.structure import Structure
from pantea.types import Array

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"


class LJPotential:
    """A simple implementation of Lennard-Jones potential."""

    def __init__(
        self,
        sigma: float,
        epsilon: float,
        r_cutoff: float,
        gradient_method: Literal["direct", "autodiff"] = "direct",
    ) -> None:
        self.sigma = jnp.array(sigma)
        self.epsilon = jnp.array(epsilon)
        self.r_cutoff = jnp.array(r_cutoff)
        self._compute_forces = self._get_force_kernel(gradient_method)

    def __call__(self, structure: Structure) -> Array:
        """Compute total potential energy."""
        return _jitted_compute_total_energy(
            LJPotentialParams(self.epsilon, self.sigma),
            structure.positions,
            structure.lattice,
            self.r_cutoff,
        )

    def compute_forces(self, structure: Structure) -> Array:
        """Compute force components for all atoms."""
        return self._compute_forces(structure)

    def _compute_forces_autodiff(self, structure: Structure) -> Array:
        return _jitted_grad_compute_total_energy(
            LJPotentialParams(self.epsilon, self.sigma),
            structure.positions,
            structure.lattice,
            self.r_cutoff,
        )

    def _compute_forces_direct(self, structure: Structure) -> Array:
        return _compute_forces(
            LJPotentialParams(self.epsilon, self.sigma),
            structure.positions,
            structure.lattice,
            self.r_cutoff,
        )

    def _get_force_kernel(self, gradient_method: str) -> Callable:
        if gradient_method == "direct":
            return self._compute_forces_direct
        elif gradient_method == "autodiff":
            return self._compute_forces_autodiff
        else:
            raise ValueError("Unknown gradient method")


class LJPotentialParams(NamedTuple):
    epsilon: Array
    sigma: Array


def _compute_pair_energies(params: LJPotentialParams, r: Array) -> Array:
    term = params.sigma / r
    term6 = term**6
    return 4.0 * params.epsilon * term6 * (term6 - 1.0)


def _compute_total_energy(
    params: LJPotentialParams,
    positions: Array,
    lattice: Optional[Array],
    r_cutoff: Array,
) -> Array:
    masks, (rij, _) = _calculate_masks_with_aux_from_structure(
        positions, r_cutoff, lattice
    )
    pair_energies = _compute_pair_energies(params, rij)
    pair_energies_inside_cutoff = jnp.where(masks, pair_energies, 0.0)
    return 0.5 * jnp.sum(pair_energies_inside_cutoff)  # type: ignore


_jitted_compute_total_energy = jax.jit(_compute_total_energy)

_grad_compute_total_energy = jax.grad(_compute_total_energy, argnums=1)

_jitted_grad_compute_total_energy = jax.jit(_grad_compute_total_energy)


def _compute_pair_forces(params: LJPotentialParams, r: Array, R: Array) -> Array:
    term = params.sigma / r
    term6 = term**6
    coefficient = -24.0 * params.epsilon / (r * r) * term6 * (2.0 * term6 - 1.0)
    return jnp.expand_dims(coefficient, axis=-1) * R


@jax.jit
def _compute_forces(
    params: LJPotentialParams,
    positions: Array,
    lattice: Optional[Array],
    r_cutoff: Array,
) -> Array:
    masks, (rij, Rij) = _calculate_masks_with_aux_from_structure(
        positions, r_cutoff, lattice
    )
    pair_forces = _compute_pair_forces(params, rij, Rij)
    pair_forces_inside_cutoff = jnp.where(
        jnp.expand_dims(masks, axis=-1),
        pair_forces,
        jnp.zeros_like(Rij),
    )
    return jnp.sum(pair_forces_inside_cutoff, axis=1)  # type: ignore
