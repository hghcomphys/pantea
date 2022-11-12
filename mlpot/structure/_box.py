import jax
import jax.numpy as jnp

Tensor = jnp.ndarray


@jax.jit
def _apply_pbc(dx: Tensor, lattice: Tensor) -> Tensor:
    """
    [Kernel]
    Apply periodic boundary condition (PBC) along x,y, and z directions.

    Make sure shifting all atoms inside the PBC box beforehand otherwise
    this method may not work as expected, see shift_inside_box().

    :param dx: Position difference
    :type dx: Tensor
    :param lattice: lattice matrix
    :type lattice: Tensor
    :return: PBC applied position
    :rtype: Tensor
    """
    l = lattice.diagonal()
    dx = jnp.where(dx > 0.5e0 * l, dx - l, dx)
    dx = jnp.where(dx < -0.5e0 * l, dx + l, dx)

    return dx
