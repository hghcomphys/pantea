from typing import Any, Protocol

from jaxip.types import Array


class DescriptorInterface(Protocol):
    """A base class for atomic environment descriptors."""

    def add(self, *args: Any, **kwargs: Any) -> None:
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Array:
        ...

    def grad(self, *args: Any, **kwargs: Any) -> Array:
        ...

    @property
    def num_descriptors(self) -> int:
        """Return number of items in the descriptor array."""
        ...

    @property
    def r_cutoff(self) -> float:
        """Return the cutoff radius of the descriptor."""
        ...
