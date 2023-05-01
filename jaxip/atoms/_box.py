import jax
import jax.numpy as jnp

from jaxip.types import Array


@jax.jit
def _apply_pbc(dx: Array, lattice: Array) -> Array:
    """Apply periodic boundary condition (PBC) along x,y, and z directions."""
    box = lattice.diagonal()
    dx = jnp.where(dx > 0.5e0 * box, dx - box, dx)
    dx = jnp.where(dx < -0.5e0 * box, dx + box, dx)

    return dx
