from .symmetry import SymmetryFunction
from .cutoff import CutoffFunction
import jax
import jax.numpy as jnp
from functools import partial

Tensor = jnp.ndarray


class RadialSymmetryFunction(SymmetryFunction):
    """
    Two body symmetry function.
    TODO: define generic **params input arguments in the base class?
    TODO: define a internal cutoff radius
    TODO: add other variant of radial symmetry functions.
    TODO: add logging when initializing each symmetry function.
    """

    def __call__(self, rij: Tensor) -> Tensor:
        raise NotImplementedError


class G1(RadialSymmetryFunction):
    """
    Plain cutoff function.
    """

    def __init__(self, cfn: CutoffFunction):
        super().__init__(cfn)

    @partial(jax.jit, static_argnums=(0,))  # FIXME
    def __call__(self, rij: Tensor) -> Tensor:
        return self.cfn(rij)


class G2(RadialSymmetryFunction):
    """
    Radial exponential term.
    """

    def __init__(self, cfn: CutoffFunction, r_shift: float, eta: float):
        self.r_shift = r_shift
        self.eta = eta
        super().__init__(cfn)

    @partial(jax.jit, static_argnums=(0,))  # FIXME
    def __call__(self, rij: Tensor) -> Tensor:
        return jnp.exp(-self.eta * (rij - self.r_shift) ** 2) * self.cfn(rij)
