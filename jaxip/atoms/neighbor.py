from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from jaxip.atoms._structure import _calculate_distances
from jaxip.logger import logger
from jaxip.pytree import BaseJaxPytreeDataClass, register_jax_pytree_node
from jaxip.types import Array


# @jax.jit
def _calculate_masks_per_atom(
    rij: Array,
    r_cutoff: Array,
) -> Array:
    """Return masks (boolean array) of a single atom inside a cutoff radius."""
    return (rij <= r_cutoff) & (rij != 0.0)


_vmap_calculate_masks: Callable = jax.vmap(
    _calculate_masks_per_atom,
    in_axes=(0, None),
)


# @jax.jit
def _calculate_masks(
    rij: Array,
    r_cutoff: Array,
) -> Array:
    """Calculate masks (boolean arrays) of multiple atoms inside a cutoff radius."""
    return _vmap_calculate_masks(rij, r_cutoff)


@jax.jit
def _calculate_masks_and_distances(
    structure,
    r_cutoff: Array,
) -> Tuple[Array, Array, Array]:
    """Calculate masks (boolean arrays) of multiple atoms inside a cutoff radius."""
    rij, Rij = _calculate_distances(
        atom_positions=structure.positions,
        neighbor_positions=structure.positions,
        lattice=structure.lattice,
    )
    masks = _calculate_masks(rij, r_cutoff)
    return masks, rij, Rij


@dataclass
class Neighbor(BaseJaxPytreeDataClass):
    """
    Create a neighbor list of atoms for structure.

    .. note::
        For MD simulations, re-neighboring the list is required every few steps.
        This is usually implemented together with defining a skin radius.
    """

    r_cutoff: float
    masks: Optional[Array] = field(default=None, init=False)
    rij: Optional[Array] = field(default=None, init=False)
    Rij: Optional[Array] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Post initialize the neighbor list."""
        logger.debug(f"Initializing {self}")
        self._assert_jit_dynamic_attributes(expected=("masks", "rij", "Rij"))
        self._assert_jit_static_attributes(expected=("r_cutoff",))

    def set_cutoff_radius(self, r_cutoff: float) -> None:
        """
        Set a given cutoff radius for the neighbor list.

        :param r_cutoff: A new cutoff radius
        :type r_cutoff: float
        """
        logger.debug(f"Setting Neighbor cutoff radius to {r_cutoff}")
        self.r_cutoff = r_cutoff

    # @jax.jit
    def update(self, structure) -> None:
        """
        Update the list of neighboring atoms.

        This approach relies on cutoff masks, which is different from conventional
        methods used to update the neighbor list (such as defining neighbor indices).
        The rationale behind this approach is that JAX executes efficiently on
        vectorized variables, offering faster performance compared to simple Python loops.
        """
        logger.debug("Updating neighbor list")
        self.masks, self.rij, self.Rij = _calculate_masks_and_distances(
            structure, jnp.atleast_1d(self.r_cutoff)
        )

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r_cutoff={self.r_cutoff})"


register_jax_pytree_node(Neighbor)
