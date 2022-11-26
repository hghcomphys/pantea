from __future__ import annotations
from .box import _apply_pbc
from typing import Tuple
import jax
import jax.numpy as jnp


@jax.jit
def _calculate_distance_per_atom(
    x_atom: jnp.ndarray,
    x_neighbors: jnp.ndarray,
    lattice: jnp.ndarray = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    [Kernel]
    Calculate a tensor of distances to all atoms existing in the structure from a given atom.
    TODO: input pbc flag, using default pbc from global configuration
    """
    dx = x_atom - x_neighbors
    if lattice is not None:
        dx = _apply_pbc(dx, lattice)

    # Fix NaN in gradient of np.linalg.norm
    # see https://github.com/google/jax/issues/3058
    is_zero = dx.sum(axis=1, keepdims=True) == 0.0
    dx_ = jnp.where(is_zero, jnp.ones_like(dx), dx)
    dist = jnp.linalg.norm(dx_, ord=2, axis=1)
    dist = jnp.where(jnp.squeeze(is_zero), 0.0, dist)

    return dist, dx


_vmap_calculate_distance = jax.vmap(
    _calculate_distance_per_atom,
    in_axes=(0, None, None),
)


@jax.jit
def _calculate_distance(
    x_atom: jnp.ndarray,
    x_neighbors: jnp.ndarray,
    lattice: jnp.ndarray = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    [Kernel]
    Calculate a tensor of distances to all atoms existing in the structure from a given atom.
    """
    # TODO: input pbc flag, using default pbc from global configuration
    return _vmap_calculate_distance(x_atom, x_neighbors, lattice)
