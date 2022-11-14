from ..logger import logger
from ..base import _Base
from ..config import dtype as _dtype
from ._box import _apply_pbc
from functools import partial
import jax
import jax.numpy as jnp

Tensor = jnp.ndarray


class Box(_Base):
    """
    Box class extract Box info from the lattice matrix.
    Currently, it only works for orthogonal lattice.
    TODO: box variables as numpy or pytorch?
    TODO: triclinic lattice
    """

    def __init__(
        self,
        lattice: Tensor = None,
        dtype: jnp.dtype = jnp.float32,  # FIXME
    ) -> None:
        """
        Initialize simulation box (super-cell).

        :param lattice: Lattice matrix (3x3 array)
        :param dtype: Data type of internal tensors which represent structure, defaults to None
        :type dtype: torch.dtype, optional
        """
        self.dtype = dtype if dtype else _dtype.FLOATX

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
    def apply_pbc(self, dx: Tensor) -> Tensor:
        """
        Apply the periodic boundary condition (PBC) on input tensor.

        :param dx: Position difference
        :type dx: Tensor
        :return: PBC applied position difference
        :rtype: Tensor
        """
        return _apply_pbc(dx, self.lattice)

    def shift_inside_box(self, x: Tensor) -> Tensor:
        """
        Shift the input atom coordinates inside the PBC simulation box.

        :param x: atom position
        :type x: Tensor
        :return: moved atom position
        :rtype: Tensor
        """
        return jnp.remainder(x, self.length)

    @property
    def lx(self) -> Tensor:
        return self.lattice[0, 0]

    @property
    def ly(self) -> Tensor:
        return self.lattice[1, 1]

    @property
    def lz(self) -> Tensor:
        return self.lattice[2, 2]

    @property
    def length(self) -> Tensor:
        return self.lattice.diagonal()