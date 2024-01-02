from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from pantea.atoms.box import _apply_pbc
from pantea.types import Array


@jax.jit
def _calculate_distances_per_atom(
    atom_position: Array,
    neighbor_positions: Array,
    lattice: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Calculate distances between a single atom and neighboring atoms."""
    dx = atom_position - neighbor_positions
    if lattice is not None:
        dx = _apply_pbc(dx, lattice)
    # Fix NaN in gradient of np.linalg.norm for zero distances
    # see https://github.com/google/jax/issues/3058
    is_zero = jnp.prod(dx == 0.0, axis=1, keepdims=True, dtype=bool)
    dx_masked = jnp.where(is_zero, 1.0, dx)
    distances = jnp.linalg.norm(dx_masked, ord=2, axis=1)  # type: ignore
    distances = jnp.where(is_zero[..., 1], 0.0, distances)
    return distances, dx


_vmap_calculate_distances: Callable = jax.vmap(
    _calculate_distances_per_atom,
    in_axes=(0, None, None),
)


@jax.jit
def _calculate_distances(
    atom_positions: Array,
    neighbor_positions: Array,
    lattice: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Calculate an array of distances between multiple atoms
    and the neighbors (using `jax.vmap`)."""
    return _vmap_calculate_distances(
        atom_positions,
        neighbor_positions,
        lattice,
    )
