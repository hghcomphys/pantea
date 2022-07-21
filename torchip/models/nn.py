from ..logger import logger
from .base import BaseModel
from typing import Tuple
from pathlib import Path
from torch import nn
import torch


class NeuralNetworkModel(BaseModel):
  """
  A neural network model which maps descriptor values to energy and force (using gradient).
  """
  def __init__(self, 
      input_size: int, 
      hidden_layers: Tuple[Tuple[int, str]],  
      output_layer:Tuple[int, str] = (1, 'l')
    ) -> None:
    super(NeuralNetworkModel, self).__init__()
    self.input_size = input_size
    self.hidden_layers = hidden_layers
    self.output_layer = output_layer

    logger.debug(f"Initializing {self}")
    self._create_network()
    
  def _create_network(self) -> None:
    """
    Create a network using provided parameters.
    """
     # Prepare stack of layers
    linear_stack = []
    in_size = self.input_size
    # Hidden layers
    for out_size, af_type in self.hidden_layers:
      linear_stack.append( nn.Linear(in_size, out_size) )
      linear_stack.append( self._get_activation_function(af_type) )
      in_size = out_size
    # Output layer
    linear_stack.append( nn.Linear(in_size, self.output_layer[0]) )
    linear_stack.append( self._get_activation_function(self.output_layer[1]) )
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
      logger.error(f"Unknown activation function type '{type_}'", exception=ValueError)

  def forward(self, x):
    return self.linear_stack(x)

  def save(self, filename: Path) -> None:
    """
    Save model weights. 
    """
    torch.save(self.state_dict(), str(filename))

  def load(self, filename: Path) -> None:
    """
    Load model weights. 
    """
    self.load_state_dict(torch.load(str(filename)))
    self.eval()

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}(input_size={self.input_size}" \
           f", hidden_layers={self.hidden_layers}, output_layer={self.output_layer})"


    