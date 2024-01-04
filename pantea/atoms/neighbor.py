from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple, Union

import jax
import jax.numpy as jnp

from pantea.atoms.distance import _calculate_distances, _calculate_distances_with_aux
from pantea.pytree import BaseJaxPytreeDataClass, register_jax_pytree_node
from pantea.types import Array


# @jax.jit
def _calculate_masks_per_atom(
    rij: Array,
    r_cutoff: Array,
) -> Array:
    """Return masks (boolean array) of a single atom inside a cutoff radius."""
    return (rij <= r_cutoff) & (rij > 0.0)


_vmap_calculate_masks: Callable = jax.vmap(
    _calculate_masks_per_atom,
    in_axes=(0, None),
)


@jax.jit
def _calculate_masks(
    rij: Array,
    r_cutoff: Array,
) -> Array:
    """Calculate masks (boolean arrays) of multiple atoms inside a cutoff radius."""
    return _vmap_calculate_masks(rij, r_cutoff)


@jax.jit
def _calculate_masks_from_structure(
    atom_positions: Array,
    r_cutoff: Array,
    lattice: Optional[Array] = None,
) -> Array:
    rij = _calculate_distances(atom_positions, atom_positions, lattice)
    return _calculate_masks(rij, r_cutoff)


@jax.jit
def _calculate_masks_with_aux_from_structure(
    atom_positions: Array,
    r_cutoff: Array,
    lattice: Optional[Array] = None,
) -> Tuple[Array, Tuple[Array, Array]]:
    rij, Rij = _calculate_distances_with_aux(atom_positions, atom_positions, lattice)
    return _calculate_masks(rij, r_cutoff), (rij, Rij)


class StructureInterface(Protocol):
    positions: Array
    lattice: Array


@dataclass
class Neighbor(BaseJaxPytreeDataClass):
    """
    Finding neighboring atoms.

    This is useful for efficiently determining the neighboring atoms within
    a specified cutoff radius. The neighbor list allows for faster calculations
    properties that depend on nearby atoms, such as computing forces, energies,
    or evaluating interatomic distances.

    The current implementation relies on cutoff masks, which is different from conventional
    methods used to update the neighbor list (such as defining neighbor indices).
    The rationale behind this approach is that JAX executes efficiently on
    vectorized variables, offering faster performance compared to simple Python loops.

    .. note::
        For MD simulations, re-neighboring the list is required every few steps.
        This is usually implemented together with defining a skin radius.
    """

    r_cutoff: Array
    masks: Array

    def __post_init__(self) -> None:
        """Post initialize the neighbor list."""
        self._assert_jit_dynamic_attributes(expected=("r_cutoff", "masks"))
        self._assert_jit_static_attributes()

    @classmethod
    def from_structure(
        cls,
        structure: StructureInterface,
        r_cutoff: float,
        with_aux: bool = False,
    ) -> Union[Neighbor, Tuple[Array, Array]]:
        rc = jnp.asarray(r_cutoff)
        if with_aux:
            masks, aux = _calculate_masks_with_aux_from_structure(
                structure.positions, rc, structure.lattice
            )
            return cls(rc, masks), aux
        else:
            masks = _calculate_masks_from_structure(
                structure.positions, rc, structure.lattice
            )
            return cls(rc, masks)

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r_cutoff={self.r_cutoff})"


register_jax_pytree_node(Neighbor)
