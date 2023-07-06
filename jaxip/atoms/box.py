from dataclasses import InitVar, dataclass
from typing import Optional

import jax
import jax.numpy as jnp

from jaxip.logger import logger
from jaxip.pytree import BaseJaxPytreeDataClass, register_jax_pytree_node
from jaxip.types import Array, Dtype, _dtype


@jax.jit
def _apply_pbc(dx: Array, lattice: Array) -> Array:
    """Apply periodic boundary condition (PBC) along x,y, and z directions."""
    box = lattice.diagonal()
    dx = jnp.where(dx > 0.5 * box, dx - box, dx)
    dx = jnp.where(dx < -0.5 * box, dx + box, dx)

    return dx


@dataclass
class Box(BaseJaxPytreeDataClass):
    """
    Simulation box which is for applying PBC
    when there lattice matrix is available.

    .. warning::
        Current implementation works only for orthogonal cells.
        No support for triclinic cells yet.
    """

    lattice: Array
    dtype: InitVar[Optional[Dtype]] = None

    def __post_init__(self, dtype: Optional[Dtype] = None) -> None:
        """Post initialize simulation box (super-cell)."""
        logger.debug(f"Initializing {self}")
        if dtype is None:
            dtype = _dtype.FLOATX
        try:
            self.lattice = jnp.array(self.lattice, dtype=dtype).reshape(3, 3)
        except ValueError:
            logger.error(
                "Unexpected lattice matrix type or dimension",
                exception=ValueError,  # type:ignore
            )

        self._assert_jit_dynamic_attributes(expected=("lattice",))

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    @jax.jit
    def apply_pbc(self, dx: Array) -> Array:
        """
        Apply the periodic boundary condition (PBC) on the input position array.

        All atoms must be positioned inside the box beforehand, otherwise
        this method may not work as expected,
        see :meth:`structure.Structure.shift_inside_box`.

        :param dx: positional difference
        :type dx: Array
        :return: PBC applied input
        :rtype: Optional[Array]
        """
        return _apply_pbc(dx, self.lattice)

    @jax.jit
    def shift_inside_box(self, positions: Array) -> Array:
        """
        Shift the input atom coordinates inside the PBC simulation box.

        :param x: atom positions
        :type x: Array
        :return: shifted atom positions
        :rtype: Array
        """
        logger.debug("Shift all atoms inside the simulation box")
        return jnp.remainder(positions, self.length)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lattice={self.lattice})"

    @property
    def lx(self) -> Array:
        """Return length of cell in x-direction."""
        return self.lattice[0, 0]

    @property
    def ly(self) -> Array:
        """Return length of cell in y-direction."""
        return self.lattice[1, 1]

    @property
    def lz(self) -> Array:
        """Return length of cell in z-direction."""
        return self.lattice[2, 2]

    @property
    def length(self) -> Array:
        """Return length of cell in x, y, and z-directions."""
        return self.lattice.diagonal()

    @property
    def volume(self) -> Array:
        """Return volume of the box."""
        return jnp.prod(self.length)


register_jax_pytree_node(Box)
