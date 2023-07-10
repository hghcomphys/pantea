import os
from typing import Tuple

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from frozendict import frozendict

from jaxip.models.nn.initializer import UniformInitializer
from jaxip.models.nn.network import NeuralNetworkModel
from jaxip.types import _dtype


class TestNeuralNetworkModel:
    nn_1: NeuralNetworkModel = NeuralNetworkModel(hidden_layers=((2, "tanh"),))
    nn_2: NeuralNetworkModel = NeuralNetworkModel(
        hidden_layers=((4, "tanh"), (8, "identity")),
        param_dtype=jnp.float64,  # type: ignore
    )
    nn_3: NeuralNetworkModel = NeuralNetworkModel(
        hidden_layers=((8, "tanh"), (16, "tanh")),
        kernel_initializer=UniformInitializer(weights_range=(-1.0, 1.0)),
    )
    nn_4: NeuralNetworkModel = NeuralNetworkModel(
        hidden_layers=((8, "tanh"), (16, "tanh")),
        kernel_initializer=UniformInitializer(weights_range=(0.0, 1.0)),
    )

    @pytest.mark.parametrize(
        "network, expected",
        [
            (
                nn_1,
                (
                    ((2, "tanh"),),
                    (1, "identity"),
                    _dtype.FLOATX,
                ),
            ),
            (
                nn_2,
                (
                    ((4, "tanh"), (8, "identity")),
                    (1, "identity"),
                    jnp.float64,
                ),
            ),
            (
                nn_3,
                (
                    ((8, "tanh"), (16, "tanh")),
                    (1, "identity"),
                    _dtype.FLOATX,
                ),
            ),
        ],
    )
    def test_attributes(
        self,
        network: NeuralNetworkModel,
        expected: Tuple,
    ) -> None:
        assert network.hidden_layers == expected[0]
        assert network.output_layer == expected[1]
        assert network.param_dtype, expected[2]

    @pytest.mark.parametrize(
        "network, expected",
        [
            (
                nn_3,
                (
                    UniformInitializer,
                    (-1.0, 1.0),
                ),
            ),
            (
                nn_4,
                (
                    UniformInitializer,
                    (0.0, 1.0),
                ),
            ),
        ],
    )
    def test_kernel_initializers(
        self,
        network: NeuralNetworkModel,
        expected: Tuple,
    ) -> None:
        assert isinstance(network.kernel_initializer, expected[0])
        assert network.kernel_initializer.weights_range == expected[1]

    @pytest.mark.parametrize(
        "network, inputs, expected",
        [
            (
                nn_3,
                (4, 5),
                (
                    frozendict(
                        {
                            "params": {
                                "layers_0": {
                                    "bias": (8,),
                                    "kernel": (5, 8),
                                },
                                "layers_2": {
                                    "bias": (16,),
                                    "kernel": (8, 16),
                                },
                                "layers_4": {
                                    "bias": (1,),
                                    "kernel": (16, 1),
                                },
                            },
                        }
                    ),
                ),
            ),
        ],
    )
    def test_network_creation(
        self,
        network: NeuralNetworkModel,
        inputs: Tuple,
        expected: Tuple,
    ) -> None:
        rng = jax.random.PRNGKey(2023)  # PRNG Key
        x = jnp.ones(shape=(inputs[0], inputs[1]))  # Dummy Input
        params = network.init(rng, x)  # type: ignore
        y = network.apply(params, x)
        assert y.shape[0] == inputs[0]  # type: ignore
        assert y.shape[1] == network.output_layer[0]  # type: ignore
        assert jax.tree_map(lambda x: x.shape, params) == expected[0]
