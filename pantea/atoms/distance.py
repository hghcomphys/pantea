from typing import Callable, Optional, Protocol, Tuple, Union

import jax
import jax.numpy as jnp

from pantea.atoms.box import _apply_pbc
from pantea.types import Array


# @jax.jit
def _calculate_distances_with_aux_per_atom(
    atom_position: Array,
    neighbor_positions: Array,
    lattice: Optional[Array] = None,
) -> Tuple[Array, Array]:
    dx = atom_position - neighbor_positions
    if lattice is not None:
        dx = _apply_pbc(dx, lattice)
    # Fix NaN in gradient of np.linalg.norm for zero distances
    # see https://github.com/google/jax/issues/3058
    is_zero = jnp.prod(dx == 0.0, axis=1, keepdims=True, dtype=bool)
    dx_masked = jnp.where(is_zero, 1.0, dx)
    # TODO: replace where with lax.cond to avoid calculating norm for all items
    distances = jnp.linalg.norm(dx_masked, ord=2, axis=1)  # type: ignore
    distances = jnp.where(is_zero[..., 1], 0.0, distances)
    return distances, dx


# @jax.jit
def _calculate_distances_per_atom(
    atom_position: Array,
    neighbor_positions: Array,
    lattice: Optional[Array] = None,
) -> Array:
    distances, _ = _calculate_distances_with_aux_per_atom(
        atom_position,
        neighbor_positions,
        lattice,
    )
    return distances


_vmap_calculate_distances_with_aux: Callable = jax.vmap(
    _calculate_distances_with_aux_per_atom,
    in_axes=(0, None, None),
)


_vmap_calculate_distances: Callable = jax.vmap(
    _calculate_distances_per_atom,
    in_axes=(0, None, None),
)


@jax.jit
def _calculate_distances(
    atom_positions: Array,
    neighbor_positions: Array,
    lattice: Optional[Array] = None,
) -> Array:
    return _vmap_calculate_distances(atom_positions, neighbor_positions, lattice)


@jax.jit
def _calculate_distances_with_aux(
    atom_positions: Array,
    neighbor_positions: Array,
    lattice: Optional[Array] = None,
) -> Tuple[Array, Array]:
    return _vmap_calculate_distances_with_aux(
        atom_positions, neighbor_positions, lattice
    )


class StructureInterface(Protocol):
    positions: Array
    lattice: Array


def calculate_distances(
    structure: StructureInterface,
    atom_indices: Optional[Array] = None,
    neighbor_indices: Optional[Array] = None,
    with_aux: bool = False,
) -> Union[Array, tuple[Array, Array]]:
    """
    Calculate distances between specific atoms and the neighboring atoms in the structure.
    This function optionally also returns the corresponding position differences.

    If atom indices are not specified, all atoms in the structure will be taken into account.
    Similarly, if neighbor indices are not provided, all neighboring atoms will be considered.

    :param structure: input structure
    :type structure: StructureInterface
    :param atom_indices: array of atom indices (zero-based)
    :type atom_indices: Optional[Array], optional
    :param neighbor_indices: indices of neighbor atoms, defaults to None
    :type neighbor_indices: Optional[Array], optional
    :param with_aux: whether returning position differences, defaults to False
    :type: bool, optional
    :return: either an array of distances or tuple of distances together and position differences
    :rtype: Union[Array, tuple[Array, Array]]
    """
    atom_positions = structure.positions
    if atom_indices is not None:
        atom_positions = structure.positions[jnp.array([atom_indices])].reshape(-1, 3)

    neighbor_positions = structure.positions
    if neighbor_indices is not None:
        neighbor_positions = structure.positions[jnp.atleast_1d(neighbor_indices)]

    distance_kernel: Callable = (
        _calculate_distances_with_aux if with_aux else _calculate_distances
    )
    return distance_kernel(
        atom_positions=atom_positions,
        neighbor_positions=neighbor_positions,
        lattice=structure.lattice,
    )
