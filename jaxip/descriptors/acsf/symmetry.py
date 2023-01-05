from abc import ABCMeta, abstractmethod
from typing import NamedTuple, Optional

from jaxip.base import _BaseJaxPytreeDataClass
from jaxip.descriptors.acsf.cutoff import CutoffFunction
from jaxip.logger import logger
from jaxip.types import Array


class SymmetryFunction(_BaseJaxPytreeDataClass, metaclass=ABCMeta):
    """
    A base class for symmetry functions.
    All symmetry functions (i.e. radial and angular) must derive from this base class.
    """

    def __init__(self, cfn: CutoffFunction) -> None:
        self.cfn: CutoffFunction = cfn
        logger.debug(repr(self))

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Array:
        pass

    @property
    def r_cutoff(self) -> float:
        return self.cfn.r_cutoff


class EnvironmentElements(NamedTuple):
    central: str
    neighbor_j: str
    neighbor_k: Optional[str] = None
