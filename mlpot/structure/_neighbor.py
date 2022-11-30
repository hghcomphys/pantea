import jax
import jax.numpy as jnp
from functools import partial


@jax.jit
def _calculate_cutoff_mask_per_atom(
    rij: jnp.ndarray,
    r_cutoff: jnp.ndarray,
) -> jnp.ndarray:
    # mask atoms only inside the cutoff radius and excluding self-counting
    return (rij <= r_cutoff) & (rij != 0.0)  # FIXME: use jnp.isclose


_vmap_calculate_neighbor_mask = jax.vmap(
    _calculate_cutoff_mask_per_atom,
    in_axes=(0, None),
)


@partial(jax.jit, static_argnums=(0,))  # FIXME
def _calculate_cutoff_mask(
    structure,
    r_cutoff: jnp.ndarray,
) -> jnp.ndarray:
    # Tensors no need to be differentiable here
    aids = jnp.arange(structure.n_atoms)  # all atoms
    rij, _ = structure.calculate_distance(aids)
    return _vmap_calculate_neighbor_mask(rij, r_cutoff)
