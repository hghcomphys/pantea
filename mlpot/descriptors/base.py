from abc import ABCMeta, abstractmethod

from mlpot.base import _BaseJaxPytreeDataClass


class Descriptor(_BaseJaxPytreeDataClass, metaclass=ABCMeta):
    """A base class for descriptors."""

    @property
    @abstractmethod
    def r_cutoff(self) -> float:
        """Return the cutoff radius of the descriptor."""
        pass
