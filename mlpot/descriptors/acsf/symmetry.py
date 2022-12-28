from abc import ABCMeta, abstractmethod

from mlpot.base import _BaseJaxPytreeDataClass
from mlpot.descriptors.acsf.cutoff import CutoffFunction
from mlpot.logger import logger
from mlpot.types import Array


class SymmetryFunction(_BaseJaxPytreeDataClass):
    """
    A base class for symmetry functions.
    All symmetry functions (i.e. radial and angular) must derive from this base class.
    """

    def __init__(self, cutoff_function: CutoffFunction) -> None:
        self.cutoff_function: CutoffFunction = cutoff_function
        logger.debug(repr(self))

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Array:
        pass

    @property
    def r_cutoff(self) -> float:
        return self.cutoff_function.r_cutoff
