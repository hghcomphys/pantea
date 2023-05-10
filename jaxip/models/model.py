from typing import Any, Protocol

from jaxip.types import Array


class ModelInterface(Protocol):
    def __call__(self, inputs: Array) -> Array:
        """Compute energy."""
        ...

    def save(self, *arg: Any, **kwargs: Any) -> None:
        """Save model weights."""
        ...

    def load(self, *arg: Any, **kwargs: Any) -> Any:
        """Load model weights."""
        ...
