from typing import Tuple, List, Callable, Mapping
from frozendict import frozendict
from flax import linen as nn
from jax.random import KeyArray
from mlpot.types import dtype as _dtype, Array, Dtype


class UniformInitializer:
    def __init__(self, weights_range: Tuple[int, int]) -> None:
        self.weights_range = weights_range
        self.initializer = nn.initializers.uniform(
            self.weights_range[1] - self.weights_range[0]
        )

    def __call__(self, rng: KeyArray, shape: Tuple[int, ...], dtype: Dtype) -> Array:
        return self.initializer(rng, shape, dtype) + self.weights_range[0]


class NeuralNetworkModel(nn.Module):  # BaseModel
    """
    A neural network model which maps descriptor values to energy and force (using gradient).
    """

    # input_size: int
    hidden_layers: Tuple[Tuple[int, str]]
    output_layer: Tuple[int, str] = (1, "l")
    weights_range: Tuple[int, int] = (-1, 1)
    param_dtype: Dtype = _dtype.FLOATX
    # TODO: add kenel initializer as input argument

    # see here https://compphysvienna.github.io/n2p2/api/neural_network.html?highlight=activation%20function
    _activation_function_map: Mapping[str, Callable] = frozendict(
        {
            "t": nn.tanh,
            "l": lambda x: x,
        }
    )

    def setup(self) -> None:
        """
        Initialize stack of Dense layers and activation functions.
        """
        self.layers = self.create_network()

    def _create_layer(self, features: int) -> nn.Dense:
        """
        Create a neural network layer and initialize weights and bias.
        See https://aiqm.github.io/torchani/examples/nnp_training.html#training-example
        """
        # TODO: add bias as input argument
        kernel_initializer = UniformInitializer(self.weights_range)
        bias_initializer = nn.initializers.zeros

        return nn.Dense(
            features,
            param_dtype=self.param_dtype,
            kernel_init=kernel_initializer,
            bias_init=bias_initializer,
        )

    def create_network(self) -> List:
        """
        Create a network using provided parameters.
        """
        # TODO: add logging
        layers: List = list()
        # Hidden layers
        for out_size, af_type in self.hidden_layers:
            layers.append(self._create_layer(out_size))
            layers.append(self._activation_function_map[af_type])
        # Output layer
        layers.append(self._create_layer(self.output_layer[0]))
        layers.append(self._activation_function_map[self.output_layer[1]])

        return layers

    def __call__(self, inputs: Array) -> Array:
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    # TODO:
    # def save(self, filename: Path) -> None:
    #     """
    #     Save model weights.
    #     """
    #     torch.save(self.state_dict(), str(filename))

    # TODO:
    # def load(self, filename: Path) -> None:
    #     """
    #     Load model weights.
    #     """
    #     self.load_state_dict(torch.load(str(filename)))
    #     self.eval()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(hidden_layers={self.hidden_layers}"
            f", output_layer={self.output_layer})"
        )
