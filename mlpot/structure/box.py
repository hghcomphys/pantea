from ..logger import logger
from ..base import _Base
from ..config import dtype as _dtype
from ._box import _apply_pbc
from typing import Optional
from functools import partial
import jax
import jax.numpy as jnp


class Box(_Base):
    """
    Box class extract Box info from the lattice matrix.
    Currently, it only works for orthogonal lattice.
    """

    # TODO: triclinic lattice

    def __init__(
        self,
        lattice: Optional[jnp.ndarray] = None,
        dtype: jnp.dtype = _dtype.FLOATX,
    ) -> None:
        """
        Initialize simulation box (super-cell).
        """
        self.dtype = dtype

        self.lattice = None
        if lattice is not None:
            try:
                self.lattice = jnp.asarray(
                    lattice,
                    dtype=self.dtype,
                ).reshape(3, 3)
            except RuntimeError:
                logger.error(
                    "Unexpected lattice matrix type or dimension", exception=ValueError
                )

        super().__init__()

    def __bool__(self):
        if self.lattice is None:
            return False
        return True

    @partial(jax.jit, static_argnums=(0,))  # FIXME
    def apply_pbc(self, dx: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the periodic boundary condition (PBC) on input tensor.

        :param dx: Position difference
        :type dx: jnp.ndarray
        :return: PBC applied position difference
        :rtype: jnp.ndarray
        """
        return _apply_pbc(dx, self.lattice)

    def shift_inside_box(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Shift the input atom coordinates inside the PBC simulation box.

        :param x: atom position
        :type x: jnp.ndarray
        :return: moved atom position
        :rtype: jnp.ndarray
        """
        return jnp.remainder(x, self.length)

    @property
    def lx(self) -> jnp.ndarray:
        return self.lattice[0, 0]

    @property
    def ly(self) -> jnp.ndarray:
        return self.lattice[1, 1]

    @property
    def lz(self) -> jnp.ndarray:
        return self.lattice[2, 2]

    @property
    def length(self) -> jnp.ndarray:
        return self.lattice.diagonal()
