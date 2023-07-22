from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

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


@jax.jit
def _shift_inside_box(positions: Array, lattice: Array) -> Array:
    """Shift the input atom position inside the PBC simulation box."""
    box = lattice.diagonal()
    return jnp.remainder(positions, box)


@dataclass
class Box(BaseJaxPytreeDataClass):
    """
    Simulation box is used to implement periodic boundary
    conditions (PBC) in the presence of a lattice matrix.

    .. warning::
        The current implementation only works for orthogonal cells
        and does not support triclinic cells.
    """

    lattice: Array

    def __post_init__(self) -> None:
        """Post initialize simulation box (super-cell)."""
        logger.debug(f"Initializing {self}")
        self._assert_jit_static_attributes()
        self._assert_jit_dynamic_attributes(expected=("lattice",))

    @classmethod
    def from_list(
        cls,
        data: List[float],
        dtype: Optional[Dtype] = None,
    ) -> Box:
        if dtype is None:
            dtype = _dtype.FLOATX
        lattice = jnp.array(data, dtype=dtype).reshape(3, 3)
        return Box(lattice)

    @jax.jit
    def apply_pbc(self, dx: Array) -> Array:
        """
        Apply the periodic boundary condition (PBC) on the provided position array.

        For this method to function correctly, it is essential that all atoms are
        initially positioned within the boundaries of the box.
        Otherwise, the results may not be as anticipated.

        :param dx: positional difference
        :type dx: Array
        :return: PBC applied input
        :rtype: Optional[Array]
        """
        return _apply_pbc(dx, self.lattice)

    @jax.jit
    def shift_inside_box(self, positions: Array) -> Array:
        """
        Adjust the coordinates of the input atoms to ensure they fall
        within the boundaries of the simulation box that implements
        periodic boundary conditions (PBC).

        :param x: atom positions
        :type x: Array
        :return: shifted atom positions
        :rtype: Array
        """
        logger.debug("Shift all atoms inside the simulation box")
        return _shift_inside_box(positions, self.lattice)

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

    @property
    def dtype(self) -> Dtype:
        return self.lattice.dtype

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(lattice={self.lattice}, dtype={self.dtype})"
        )


register_jax_pytree_node(Box)
