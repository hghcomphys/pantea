from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import jax
import jax.numpy as jnp

from pantea.logger import logger
from pantea.pytree import BaseJaxPytreeDataClass, register_jax_pytree_node
from pantea.types import Array, Dtype, default_dtype


@dataclass
class Box(BaseJaxPytreeDataClass):
    """
    Simulation box is used to apply periodic boundary
    conditions (PBC) in the presence of a lattice info.

    .. warning::
        The current implementation only works for orthogonal cells
        and does not support triclinic cells.
    """

    lattice: Array

    def __post_init__(self) -> None:
        self._assert_jit_static_attributes()
        self._assert_jit_dynamic_attributes(expected=("lattice",))

    @classmethod
    def from_list(
        cls,
        data: Sequence[float],
        dtype: Optional[Dtype] = None,
    ) -> Box:
        logger.debug(f"Initializing {cls.__name__} from list")
        if dtype is None:
            dtype = default_dtype.FLOATX
        lattice = jnp.array(data, dtype=dtype).reshape(3, 3)
        return Box(lattice)

    @jax.jit
    def apply_pbc(self, dx: Array) -> Array:
        """
        Apply periodic boundary condition (PBC) on the atom positions.

        For this method to function correctly, it is essential that all atoms are
        initially positioned within the boundaries of the box.
        Otherwise, the results may not be as anticipated. This could happen for
        when example time step is too large.

        :param dx: positional differences
        :type dx: Array
        :return: PBC applied position differences
        :rtype: Array
        """
        return _apply_pbc(dx, self.lattice)

    @jax.jit
    def wrap_into_box(self, positions: Array) -> Array:
        """
        Adjust the coordinates of the atoms to ensure they fall
        within the boundaries of the simulation box that has
        periodic boundary conditions (PBC).

        :param positions: atom positions
        :type positions: Array
        :return: wrapped atom positions using the PBC.
        :rtype: Array
        """
        logger.debug("Shift all atoms within the simulation box")
        return _wrap_into_box(positions, self.lattice)

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
        """Return calculated volume of the box."""
        return jnp.prod(self.length)

    @property
    def dtype(self) -> Dtype:
        return self.lattice.dtype

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lattice={self.lattice}, dtype={self.dtype})"


def _apply_pbc(dx: Array, lattice: Array) -> Array:
    """Apply periodic boundary condition (PBC) along x,y, and z directions."""
    box = lattice.diagonal()
    dx = jnp.where(dx > 0.5 * box, dx - box, dx)
    dx = jnp.where(dx < -0.5 * box, dx + box, dx)
    return dx


_jitted_apply_pbc = jax.jit(_apply_pbc)


def _wrap_into_box(positions: Array, lattice: Array) -> Array:
    """Wrap atoms back into the simulation box using periodic boundary condition."""
    box = lattice.diagonal()
    return jnp.remainder(positions, box)


_jitted_wrap_into_box = jax.jit(_wrap_into_box)


register_jax_pytree_node(Box)
