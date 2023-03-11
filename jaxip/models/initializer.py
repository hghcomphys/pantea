from typing import Tuple

from flax import linen as nn
from jax.random import KeyArray

from jaxip.types import Array, Dtype


class UniformInitializer:
    """Custom uniform initializer for the FLAX model."""

    def __init__(self, weights_range: Tuple[float, float]) -> None:
        self.weights_range = weights_range
        self.initializer = nn.initializers.uniform(
            self.weights_range[1] - self.weights_range[0]
        )

    def __call__(self, rng: KeyArray, shape: Tuple[int, ...], dtype: Dtype) -> Array:
        return self.initializer(rng, shape, dtype) + self.weights_range[0]
