import jax
import jax.numpy as jnp

from jaxip.types import Array


@jax.jit
def _calculate_cutoff_mask_per_atom(
    rij: Array,
    r_cutoff: Array,
) -> Array:
    """Create mask (boolean array) of atoms inside the cutoff radius and excluding self-counting."""
    return (rij <= r_cutoff) & (rij != 0.0)


_vmap_calculate_neighbor_mask = jax.vmap(
    _calculate_cutoff_mask_per_atom,
    in_axes=(0, None),
)


@jax.jit
def _calculate_cutoff_mask(
    structure,
    r_cutoff: Array,
) -> Array:
    """Calculate mask (boolean arrays) of neighboring atoms for multiple atoms (using `jax.vmap`)."""
    atom_index: Array = jnp.arange(structure.natoms)  # all atoms
    rij, _ = structure.calculate_distance(atom_index)
    return _vmap_calculate_neighbor_mask(rij, r_cutoff)
