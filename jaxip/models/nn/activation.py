import jax.numpy as jnp
from flax import linen as nn
from frozendict import frozendict

from jaxip.types import Array


def identity(x: Array) -> Array:
    return x


def tanh(x: Array) -> Array:
    return nn.tanh(x)


def logistic(x: Array) -> Array:
    return 1.0 / (1.0 + jnp.exp(-x))


def softplus(x: Array) -> Array:
    return nn.softplus(x)


def relu(x: Array) -> Array:
    return nn.relu(x)


def gaussian(x: Array) -> Array:
    return jnp.exp(-0.5 * x**2)


def cos(x: Array) -> Array:
    return jnp.cos(x)


def revlogistic(x: Array) -> Array:
    return 1.0 - 1.0 / (1.0 + jnp.exp(-x))


def exp(x: Array) -> Array:
    return jnp.exp(-x)


def harmonic(x: Array) -> Array:
    return x * x


# see here https://compphysvienna.github.io/n2p2/api/neural_network.html?highlight=activation%20function
_activation_function_map: frozendict = frozendict(
    {
        "identity": identity,
        "tanh": nn.tanh,
        "logistic": logistic,
        "softplus": nn.softplus,
        "relu": nn.relu,
        "gaussian": gaussian,
        "cos": cos,
        "exp": exp,
        "harmonic": harmonic,
    }
)
