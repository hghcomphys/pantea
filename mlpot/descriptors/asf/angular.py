from .cutoff import CutoffFunction
from .symmetry import SymmetryFunction
import jax
import jax.numpy as jnp
from functools import partial

Tensor = jnp.ndarray


class AngularSymmetryFunction(SymmetryFunction):
    """
    Three body symmetry function.
    TODO: add other variant of angular symmetry functions (see N2P2 documentation).
    """

    def __call__(self, rij: Tensor, rik: Tensor, rjk: Tensor, cost: Tensor) -> Tensor:
        raise NotImplementedError


class G3(AngularSymmetryFunction):
    """
    Angular symmetry function.
    """

    def __init__(
        self,
        cfn: CutoffFunction,
        eta: float,
        zeta: float,
        lambda0: float,
        r_shift: float,
    ) -> None:
        self.eta = eta
        self.zeta = zeta
        self.lambda0 = lambda0
        self.r_shift = r_shift
        self._scale_factor = 2.0 ** (1.0 - self.zeta)
        super().__init__(cfn)

    @partial(jax.jit, static_argnums=(0,))  # FIXME
    def __call__(self, rij: Tensor, rik: Tensor, rjk: Tensor, cost: Tensor) -> Tensor:
        # TODO: r_shift, define params argument instead
        return (
            self._scale_factor
            * jnp.power(1 + self.lambda0 * cost, self.zeta)
            * jnp.exp(-self.eta * (rij**2 + rik**2 + rjk**2))
            * self.cfn(rij)
            * self.cfn(rik)
            * self.cfn(rjk)
        )


class G9(G3):  # AngularSymmetryFunction
    """
    Modified angular symmetry function.
    Ref -> J. Behler, J. Chem. Phys. 134, 074106 (2011)
    """

    @partial(jax.jit, static_argnums=(0,))  # FIXME
    def __call__(self, rij: Tensor, rik: Tensor, rjk: Tensor, cost: Tensor) -> Tensor:
        # TODO: r_shift, define params argument instead
        return (
            self._scale_factor
            * jnp.pow(1 + self.lambda0 * cost, self.zeta)
            * jnp.exp(-self.eta * (rij**2 + rik**2))
            * self.cfn(rij)
            * self.cfn(rik)
        )
