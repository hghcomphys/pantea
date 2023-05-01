from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from jaxip.atoms.box import _apply_pbc
from jaxip.types import Array


@jax.jit
def _calculate_distance_per_atom(
    atom_position: Array,
    neighbor_position: Array,
    lattice: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Calculate an array of distances between a single atom and neighboring atoms."""
    dx: Array = atom_position - neighbor_position
    if lattice is not None:
        dx = _apply_pbc(dx, lattice)

    # Fix NaN in gradient of np.linalg.norm
    # see https://github.com/google/jax/issues/3058
    is_zero = dx.sum(axis=1, keepdims=True) == 0.0
    dx_ = jnp.where(is_zero, jnp.ones_like(dx), dx)
    dist = jnp.linalg.norm(dx_, ord=2, axis=1)
    dist = jnp.where(jnp.squeeze(is_zero), 0.0, dist)

    return dist, dx  # type: ignore


_vmap_calculate_distance: Callable = jax.vmap(
    _calculate_distance_per_atom,
    in_axes=(0, None, None),
)


@jax.jit
def _calculate_distance(
    atom_position: Array,
    neighbor_position: Array,
    lattice: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Calculate an array of distances between multiple atoms and the neighbors (using `jax.vmap`)."""
    return _vmap_calculate_distance(atom_position, neighbor_position, lattice)
