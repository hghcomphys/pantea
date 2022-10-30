from ...logger import logger
from ...base import _Base
from .cutoff import CutoffFunction
import jax
import jax.numpy as jnp
from jax import grad, vmap
from functools import partial

Tensor = jnp.ndarray


class SymmetryFunction(_Base):
    """
    A base class for symmetry functions.
    All symmetry functions (i.e. radial and angular) must derive from this base class.
    """

    def __init__(self, cfn: CutoffFunction):
        self.cfn = cfn
        logger.debug(repr(self))

    def __call__(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))  # FIXME
    def grad(self, *args, **kwargs):
        try:
            return vmap(grad(self), in_axes=0)(*args)
        except ValueError:
            pass
        return grad(self)(*args)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            + ", ".join(
                ["{}={!r}".format(k, v) for k, v in self.__dict__.items() if "_"]
            )
            + ")"
        )

    @property
    def r_cutoff(self) -> float:
        return self.cfn.r_cutoff
