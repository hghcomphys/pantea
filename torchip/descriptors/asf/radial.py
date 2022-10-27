from ...logger import logger
from .symmetry import SymmetryFunction
from .cutoff import CutoffFunction
import torch
from torch import Tensor


class RadialSymmetryFunction(SymmetryFunction):
    """
    Two body symmetry function.
    TODO: define generic **params input arguments in the base class?
    TODO: add __call__() method?
    TODO: define a internal cutoff radius
    TODO: add other variant of radial symmetry functions.
    TODO: add logging when initializing each symmetry function.
    """

    def kernel(self, rij: Tensor) -> Tensor:
        raise NotImplementedError


class G1(RadialSymmetryFunction):
    """
    Plain cutoff function.
    """

    def __init__(self, cfn: CutoffFunction):
        super().__init__(cfn)

    def kernel(self, rij: Tensor) -> Tensor:
        return self.cfn(rij)


@torch.jit.script
def _G2_kernel(rij: Tensor, eta: float, r_shift: float) -> Tensor:
    return torch.exp(-eta * (rij - r_shift) ** 2)


class G2(RadialSymmetryFunction):
    """
    Radial exponential term.
    """

    def __init__(self, cfn: CutoffFunction, r_shift: float, eta: float):
        self.r_shift = r_shift
        self.eta = eta
        self._params = [self.eta, self.r_shift]
        super().__init__(cfn)

    def kernel(self, rij: Tensor) -> Tensor:
        return _G2_kernel(rij, self.eta, self.r_cutoff) * self.cfn(rij)
        # return radial_cpp.g2_kernel(rij, self._params) * self.cfn(rij)
