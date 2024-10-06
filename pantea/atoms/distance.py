from __future__ import annotations

from typing import Optional, Protocol, Tuple, Union

import jax.numpy as jnp
from jax import jit, vmap

from pantea.atoms.box import _apply_pbc
from pantea.types import Array


class StructureInterface(Protocol):
    positions: Array
    lattice: Array


def calculate_distances(
    structure: StructureInterface,
    atom_index: Optional[Array] = None,
    neighbor_atom_index: Optional[Array] = None,
    with_aux: bool = False,
) -> Union[Array, tuple[Array, Array]]:
    """
    Calculate distances between specific atoms and the neighboring atoms in the structure.
    This function optionally also returns the corresponding position differences.

    If atom indices are not specified, all atoms in the structure will be taken into account.
    Similarly, if neighbor indices are not provided, all neighboring atoms will be considered.

    :param structure: input structure
    :type structure: StructureInterface
    :param atom_index: array of atom indices (zero-based)
    :type atom_index: Optional[Array], optional
    :param neighbor_atom_index: array of indices of neighbor atoms (zero-based), defaults to None
    :type neighbor_atom_index: Optional[Array], optional
    :param with_aux: whether returning position differences, defaults to False
    :type: bool, optional
    :return: either an array of distances or tuple of distances together and position differences
    :rtype: Union[Array, tuple[Array, Array]]
    """
    atom_positions = structure.positions
    if atom_index is not None:
        index = jnp.atleast_1d(atom_index)
        atom_positions = structure.positions[index]

    neighbor_atom_positions = structure.positions
    if neighbor_atom_index is not None:
        index = jnp.atleast_1d(neighbor_atom_index)
        neighbor_atom_positions = structure.positions[index]

    kernel = (
        _jitted_calculate_distances_with_aux
        if with_aux
        else _jitted_calculate_distances
    )
    return kernel(
        atom_positions,
        neighbor_atom_positions,
        structure.lattice,
    )  # type: ignore


def _calculate_distances_with_aux_per_atom(
    atom_position: Array,
    neighbor_atom_positions: Array,
    lattice: Optional[Array] = None,
) -> Tuple[Array, Array]:
    dx = atom_position - neighbor_atom_positions
    if lattice is not None:
        dx = _apply_pbc(dx, lattice)
    # Fix NaN in gradient of np.linalg.norm for zero distances
    # see https://github.com/google/jax/issues/3058
    is_zero = jnp.prod(dx == 0.0, axis=1, keepdims=True, dtype=bool)
    dx_masked = jnp.where(is_zero, 1.0, dx)
    # TODO: replace where with lax.cond to avoid calculating norm for all items
    distances = jnp.linalg.norm(dx_masked, ord=2, axis=1)
    distances = jnp.where(is_zero[..., 1], 0.0, distances)
    return distances, dx


_calculate_distances_with_aux = vmap(
    _calculate_distances_with_aux_per_atom,
    in_axes=(0, None, None),
)

_jitted_calculate_distances_with_aux = jit(_calculate_distances_with_aux)


def _calculate_distances_per_atom(
    atom_position: Array,
    neighbor_atom_positions: Array,
    lattice: Optional[Array] = None,
) -> Array:
    distances, _ = _calculate_distances_with_aux_per_atom(
        atom_position, neighbor_atom_positions, lattice
    )
    return distances


_calculate_distances = vmap(
    _calculate_distances_per_atom,
    in_axes=(0, None, None),
)

_jitted_calculate_distances = jit(_calculate_distances)
