import jax.numpy as jnp
from typing import Tuple, List, Dict, Callable
from frozendict import frozendict
from flax import linen as nn
from mlpot.types import Array


class NeuralNetworkModel(nn.Module):  # BaseModel
    """
    A neural network model which maps descriptor values to energy and force (using gradient).
    """

    # input_size: int
    hidden_layers: Tuple[Tuple[int, str]]
    output_layer: Tuple[int, str] = (1, "l")
    weights_range: Tuple[int, int] = None
    param_dtype: jnp.dtype = jnp.float32  # FIXME:

    # see here https://compphysvienna.github.io/n2p2/api/neural_network.html?highlight=activation%20function
    _activation_function_map: Dict[str, Callable] = frozendict(
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
        # FIXME: init weights using self.weights_range
        # TODO: add bias as input argument
        return nn.Dense(features, param_dtype=self.param_dtype)

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
