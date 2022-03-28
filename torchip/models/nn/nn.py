
from ...logger import logger
from ..base import BaseModel
from torch import nn
from typing import Tuple


class NeuralNetworkModel(BaseModel):
  """
  A neural network model which map descriptor values to energy and force (derivative).
  """
  def __init__(self, 
    input_size: int, 
    hidden_layers: Tuple[Tuple[int, str]], 
    output_layer:Tuple[int, str] = (1, 'l')):
    super(NeuralNetworkModel, self).__init__()

    # Prepare stack of layers
    linear_stack = []
    in_size = input_size
    # Hidden layers
    for out_size, af_type in hidden_layers:
      linear_stack.append( nn.Linear(in_size, out_size) )
      linear_stack.append( self._get_activation_function(af_type) )
      in_size = out_size
    # Output layer
    linear_stack.append( nn.Linear(in_size, output_layer[0]) )
    linear_stack.append( self._get_activation_function(output_layer[1]) )
    # Build a sequential model
    self.linear_stack = nn.Sequential(*linear_stack)
    # TODO: add logging

  def _get_activation_function(self, type_: str):
    """
    Return the corresponding activation function. 
    See here https://compphysvienna.github.io/n2p2/api/neural_network.html?highlight=activation%20function
    """
    if type_ == "t":
      return nn.Tanh()
    elif type_ == "l":
      return nn.Identity()  # No activation function
    else:
      msg = f"Unknown activation function type '{type_}'"
      logger.error(msg)
      raise ValueError(msg)

  def forward(self, x):
    return self.linear_stack(x)


    