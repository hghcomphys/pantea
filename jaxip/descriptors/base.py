from abc import ABCMeta, abstractmethod

from jaxip.base import _BaseJaxPytreeDataClass
from jaxip.types import Array


class Descriptor(_BaseJaxPytreeDataClass, metaclass=ABCMeta):
    """A base class for atomic environment descriptors."""

    @abstractmethod
    def add(self, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Array:
        ...

    @abstractmethod
    def grad(self, *args, **kwargs) -> Array:
        ...

    @property
    @abstractmethod
    def num_descriptors(self) -> int:
        """Return number of items in the descriptor array."""
        ...

    @property
    @abstractmethod
    def r_cutoff(self) -> float:
        """Return the cutoff radius of the descriptor."""
        ...
