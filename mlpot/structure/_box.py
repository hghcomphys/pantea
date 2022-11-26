import jax
import jax.numpy as jnp


@jax.jit
def _apply_pbc(dx: jnp.ndarray, lattice: jnp.ndarray) -> jnp.ndarray:
    """
    [Kernel]
    Apply periodic boundary condition (PBC) along x,y, and z directions.

    Make sure shifting all atoms inside the PBC box beforehand otherwise
    this method may not work as expected, see shift_inside_box().

    :param dx: Position difference
    :type dx: jnp.ndarray
    :param lattice: lattice matrix
    :type lattice: jnp.ndarray
    :return: PBC applied position
    :rtype: jnp.ndarray
    """
    l = lattice.diagonal()
    dx = jnp.where(dx > 0.5e0 * l, dx - l, dx)
    dx = jnp.where(dx < -0.5e0 * l, dx + l, dx)

    return dx
