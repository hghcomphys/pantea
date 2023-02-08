from dataclasses import field
from typing import Callable, List, Optional, Tuple

from flax import linen as nn

from jaxip.models.activation import activation_function
from jaxip.types import Array, Dtype
from jaxip.types import dtype as _dtype

# FIXME: add kernel initializer as input argument
# UniformInitializer(self.weights_range)
# weights_range: Tuple[int, int] = (-1, 1)


class NeuralNetworkModel(nn.Module):
    """Neural network model that outputs energy."""

    # input_size: int
    hidden_layers: Tuple[Tuple[int, str], ...]
    output_layer: Tuple[int, str] = (1, "identity")
    param_dtype: Optional[Dtype] = field(default_factory=lambda: _dtype.FLOATX)
    kernel_initializer: Callable = nn.initializers.lecun_normal()
    # bias_initializer: Callable = nn.initializers.zeros

    def setup(self) -> None:
        """Initialize neural network model."""
        self.layers: List = self.create_network()

    def create_layer(self, features: int) -> nn.Dense:
        """
        Create a dense layer and initialize the weights and biases
        (see `here <https://aiqm.github.io/torchani/examples/nnp_training.html#training-example>`_).
        """
        return nn.Dense(
            features,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_initializer,
            bias_init=nn.initializers.zeros,  # self.bias_initializer,
        )

    def create_network(self) -> List:
        """Create a neural network as stack of dense layers and activation functions."""
        # TODO: add logging
        layers: List = list()
        # Hidden layers
        for out_size, af_type in self.hidden_layers:
            layers.append(self.create_layer(out_size))
            layers.append(activation_function[af_type])
        # Output layer
        layers.append(self.create_layer(self.output_layer[0]))
        layers.append(activation_function[self.output_layer[1]])
        return layers

    def __call__(self, inputs: Array) -> Array:
        """Compute energy."""
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(hidden_layers={self.hidden_layers}"
            f", output_layer={self.output_layer}"
            # f", param_dtype={self.param_dtype.dtype})"  # type: ignore
        )

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
