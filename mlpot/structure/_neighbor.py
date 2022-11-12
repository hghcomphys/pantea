import jax
import jax.numpy as jnp
from functools import partial

Tensor = jnp.ndarray


@jax.jit
def _calculate_cutoff_mask_per_atom(
    aid: Tensor,
    rij: Tensor,
    r_cutoff: Tensor,
) -> Tensor:
    # Select atoms inside cutoff radius
    mask = rij <= r_cutoff
    # exclude self-counting
    mask = mask.at[aid].set(False)
    return mask


_vmap_calculate_neighbor_mask = jax.vmap(
    _calculate_cutoff_mask_per_atom,
    in_axes=(0, 0, None),
)


@partial(jax.jit, static_argnums=(0,))  # FIXME
def _calculate_cutoff_mask(
    structure,
    r_cutoff: Tensor,
) -> Tensor:
    # Tensors no need to be differentiable here
    aids = jnp.arange(structure.natoms)  # all atoms
    rij, _ = structure.calculate_distance(aids)
    return _vmap_calculate_neighbor_mask(aids, rij, r_cutoff)
