from ...logger import logger
from ...base import _Base
from torch import Tensor
import torch


@torch.jit.script
def _apply_cutoff(cfn_r: Tensor, r: Tensor, r_cutoff: float):
    return torch.where(r < r_cutoff, cfn_r, torch.zeros_like(r))


@torch.jit.script
def _hard(r: Tensor, r_cutoff: float):
    cfn_r: Tensor = torch.ones_like(r)
    return _apply_cutoff(cfn_r, r, r_cutoff)


@torch.jit.script
def _tanhu(r: Tensor, r_cutoff: float):
    cfn_r: Tensor = torch.tanh(1.0 - r / r_cutoff) ** 3
    return _apply_cutoff(cfn_r, r, r_cutoff)


@torch.jit.script
def _tanh(r: Tensor, r_cutoff: float, _TANH_PRE: float):
    cfn_r: Tensor = _TANH_PRE * torch.tanh(1.0 - r / r_cutoff) ** 3
    return _apply_cutoff(cfn_r, r, r_cutoff)


@torch.jit.script
def _cos(r: Tensor, r_cutoff: float):
    cfn_r: Tensor = 0.5 * (torch.cos(torch.pi * r / r_cutoff) + 1.0)
    return _apply_cutoff(cfn_r, r, r_cutoff)


@torch.jit.script
def _exp(r: Tensor, r_cutoff: float):
    cfn_r: Tensor = torch.exp(1.0 - 1.0 / (1.0 - (r / r_cutoff) ** 2))
    return _apply_cutoff(cfn_r, r, r_cutoff)


@torch.jit.script
def _poly1(r: Tensor, r_cutoff: float):
    cfn_r: Tensor = (2.0 * r - 3.0) * r**2 + 1.0
    return _apply_cutoff(cfn_r, r, r_cutoff)


@torch.jit.script
def _poly2(r: Tensor, r_cutoff: float):
    cfn_r: Tensor = ((15.0 - 6.0 * r) * r - 10) * r**3 + 1.0
    return _apply_cutoff(cfn_r, r, r_cutoff)


class CutoffFunction(_Base):
    """
    This class contains different cutoff functions used for ASF descriptor.
    TODO: optimization
    TODO: add logger
    TODO: add poly 3 & 4 funcions
    See N2P2 -> https://compphysvienna.github.io/n2p2/api/cutoff_functions.html?highlight=cutoff#
                https://compphysvienna.github.io/n2p2/topics/keywords.html?highlight=cutoff_type
    TODO: define inv_r_cutoff
    """

    _TANH_PRE = ((torch.e + 1 / torch.e) / (torch.e - 1 / torch.e)) ** 3

    def __init__(self, r_cutoff: float, cutoff_type: str = "tanh"):
        self.r_cutoff = r_cutoff
        self.cutoff_type = cutoff_type.lower()
        # Set cutoff type function
        try:
            self.cutoff_function = getattr(self, f"{self.cutoff_type}")
        except AttributeError:
            logger.error(
                f"'{self.__class__.__name__}' has no cutoff function '{self.cutoff_type}'",
                exception=NotImplementedError,
            )

    def __call__(self, r: Tensor) -> Tensor:
        return self.cutoff_function(r)

    def hard(self, r: Tensor) -> Tensor:
        return _hard(r, self.r_cutoff)

    def tanhu(self, r: Tensor) -> Tensor:
        return _tanhu(r, self.r_cutoff)

    def tanh(self, r: Tensor) -> Tensor:
        return _tanh(r, self.r_cutoff, self._TANH_PRE)

    def cos(self, r: Tensor) -> Tensor:
        return _cos(r, self.r_cutoff)

    def exp(self, r: Tensor) -> Tensor:
        return _exp(r, self.r_cutoff)

    def poly1(self, r: Tensor) -> Tensor:
        return _poly1(r, self.r_cutoff)

    def poly2(self, r: Tensor) -> Tensor:
        return _poly2(r, self.r_cutoff)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r_cutoff={self.r_cutoff}, cutoff_type='{self.cutoff_type}')"
