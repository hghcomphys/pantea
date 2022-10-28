from .cutoff import CutoffFunction
from .symmetry import SymmetryFunction
from torch import Tensor
import torch

# import angular_cpp


class AngularSymmetryFunction(SymmetryFunction):
    """
    Three body symmetry function.
    TODO: add other variant of angular symmetry functions (see N2P2 documentation).
    """

    def kernel(self, rij: Tensor, rik: Tensor, rjk: Tensor, cost: Tensor) -> Tensor:
        raise NotImplementedError


@torch.jit.script
def _G3_kernel(
    rij: Tensor,
    rik: Tensor,
    rjk: Tensor,
    cost: Tensor,
    eta: float,
    zeta: float,
    lambda0: float,
    r_shift: float,
    _scale_factor: float,
) -> Tensor:
    # TODO: r_shift, define params argument instead
    return (
        _scale_factor
        * torch.pow(1 + lambda0 * cost, zeta)
        * torch.exp(-eta * (rij**2 + rik**2 + rjk**2))
    )


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

    def kernel(self, rij: Tensor, rik: Tensor, rjk: Tensor, cost: Tensor) -> Tensor:
        return (
            _G3_kernel(
                rij,
                rik,
                rjk,
                cost,
                self.eta,
                self.zeta,
                self.lambda0,
                self.r_shift,
                self._scale_factor,
            )
            * self.cfn(rij)
            * self.cfn(rik)
            * self.cfn(rjk)
        )


@torch.jit.script
def _G9_kernel(
    rij: Tensor,
    rik: Tensor,
    cost: Tensor,
    eta: float,
    zeta: float,
    lambda0: float,
    r_shift: float,
    _scale_factor: float,
) -> Tensor:
    # TODO: r_shift, define params argument instead
    return (
        _scale_factor
        * torch.pow(1 + lambda0 * cost, zeta)
        * torch.exp(-eta * (rij**2 + rik**2))
    )


class G9(G3):  # AngularSymmetryFunction
    """
    Modified angular symmetry function.
    Ref -> J. Behler, J. Chem. Phys. 134, 074106 (2011)
    """

    def kernel(self, rij: Tensor, rik: Tensor, rjk: Tensor, cost: Tensor) -> Tensor:
        return (
            _G9_kernel(
                rij,
                rik,
                cost,
                self.eta,
                self.zeta,
                self.lambda0,
                self.r_shift,
                self._scale_factor,
            )
            * self.cfn(rij)
            * self.cfn(rik)
        )
