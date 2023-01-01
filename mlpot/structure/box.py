from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp

from mlpot.base import _BaseJaxPytreeDataClass, register_jax_pytree_node
from mlpot.logger import logger
from mlpot.structure._box import _apply_pbc
from mlpot.types import Array, Dtype
from mlpot.types import dtype as _dtype


@dataclass
class Box(_BaseJaxPytreeDataClass):
    """
    Simulation box which is responsible for applying PBCs
    when there are available lattice info.

    .. warning::
        Current implementation works only for orthogonal cells.
        No support for triclinic cells yet.
    """

    # Array type attributes must be define first (with type hint)
    lattice: Optional[Array] = None
    dtype: Dtype = _dtype.FLOATX

    def __pos_init__(self) -> None:
        """Post initialize simulation box (super-cell)."""
        if self.lattice is not None:
            try:
                self.lattice = jnp.asarray(
                    self.lattice,
                    dtype=self.dtype,
                ).reshape(3, 3)
            except RuntimeError:
                logger.error(
                    "Unexpected lattice matrix type or dimension",
                    exception=ValueError,  # type:ignore
                )
        # super().__init__()

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    def __bool__(self) -> bool:
        """Check whether the box instance with lattice info makes sense."""
        if self.lattice is None:
            return False
        return True

    @jax.jit
    def apply_pbc(self, dx: Array) -> Array:
        """
        Apply the periodic boundary condition (PBC) on the input position array.

        All atoms must be positioned inside the box beforehand, otherwise
        this method may not work as expected, see :meth:`structure.Structure.shift_inside_box`.

        :param dx: positional difference
        :type dx: Array
        :return: PBC applied input
        :rtype: Optional[Array]
        """
        if self.lattice is not None:
            return _apply_pbc(dx, self.lattice)
        return dx

    @jax.jit
    def shift_inside_box(self, x: Array) -> Array:
        """
        Shift the input atom coordinates inside the PBC simulation box.

        :param x: atom positions
        :type x: Array
        :return: shifted atom positions
        :rtype: Array
        """
        if self.length is not None:
            logger.debug("Shift all atoms inside the simulation box")
            return jnp.remainder(x, self.length)
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lattice={self.lattice})"

    @property
    def lx(self) -> Optional[Array]:
        """Return length of cell in x-direction."""
        if self.lattice is not None:
            return self.lattice[0, 0]
        return None

    @property
    def ly(self) -> Optional[Array]:
        """Return length of cell in y-direction."""
        if self.lattice is not None:
            return self.lattice[1, 1]
        return None

    @property
    def lz(self) -> Optional[Array]:
        """Return length of cell in z-direction."""
        if self.lattice is not None:
            return self.lattice[2, 2]
        return None

    @property
    def length(self) -> Optional[Array]:
        """Return length of cell in x, y, and z-directions."""
        if self.lattice is not None:
            return self.lattice.diagonal()
        return None


register_jax_pytree_node(Box)
