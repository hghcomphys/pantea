from .cutoff import CutoffFunction
from .symmetry import SymmetryFunction
from torch import Tensor
import math
import torch

# import angular_cpp


class AngularSymmetryFunction(SymmetryFunction):
    """
    Three body symmetry function.
    TODO: add other variant of angular symmetry functions (see N2P2 documentation).
    """

    def kernel(self, rij: Tensor, rik: Tensor, rjk: Tensor, cost: Tensor) -> Tensor:
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
        self._scale_factor = math.pow(2.0, 1.0 - self.zeta)
        super().__init__(cfn)
        # self._params = [self.eta, self.zeta, self.lambda0, self.r_shift, self._scale_factor]

    def kernel(self, rij: Tensor, rik: Tensor, rjk: Tensor, cost: Tensor) -> Tensor:
        res = (
            self._scale_factor
            * torch.pow(1 + self.lambda0 * cost, self.zeta)
            * torch.exp(-self.eta * (rij**2 + rik**2 + rjk**2))
        )  # TODO: r_shift
        # res = angular_cpp.g3_kernel(rij, rik, rjk, cost, self._params)
        return res * self.cfn(rij) * self.cfn(rik) * self.cfn(rjk)


class G9(G3):  # AngularSymmetryFunction
    """
    Modified angular symmetry function.
    Ref -> J. Behler, J. Chem. Phys. 134, 074106 (2011)
    """

    # def __init__(self, cfn: CutoffFunction, eta: float, zeta: float, lambda0: float, r_shift: float) -> None:
    #   self.eta = eta
    #   self.zeta = zeta
    #   self.lambda0 = lambda0
    #   self.r_shift = r_shift
    #   self._scale_factor = math.pow(2.0, 1.0-self.zeta)
    #   self._params = [self.eta, self.zeta, self.lambda0, self.r_shift, self._scale_factor]
    #   super().__init__(cfn)

    def kernel(self, rij: Tensor, rik: Tensor, rjk: Tensor, cost: Tensor) -> Tensor:
        res = (
            self._scale_factor
            * torch.pow(1 + self.lambda0 * cost, self.zeta)
            * torch.exp(-self.eta * (rij**2 + rik**2))
        )  # TODO: r_shift
        # res = angular_cpp.g9_kernel(rij, rik, rjk, cost, self._params)
        return res * self.cfn(rij) * self.cfn(rik)
