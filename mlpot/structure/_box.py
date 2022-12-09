import jax
import jax.numpy as jnp
from mlpot.types import Array


@jax.jit
def _apply_pbc(dx: Array, lattice: Array) -> Array:
    """
    [Kernel]
    Apply periodic boundary condition (PBC) along x,y, and z directions.

    Make sure shifting all atoms inside the PBC box beforehand otherwise
    this method may not work as expected, see shift_inside_box().
    """
    l = lattice.diagonal()
    dx = jnp.where(dx > 0.5e0 * l, dx - l, dx)
    dx = jnp.where(dx < -0.5e0 * l, dx + l, dx)

    return dx
