import torch
import numpy as np
from torch import Tensor


def gradient(y: Tensor, x: Tensor, grad_outputs=None) -> Tensor:
  """
  An utility function that computes derivative of input *y* tensor respect to input *x* at grad_outputs.

  Args:
      y (Tensor): output of the differentiated function
      x (Tensor):  Input w.r.t. which the gradient will be returned 
      grad_outputs (Tensor, optional): The “vector” in the vector-Jacobian product. Defaults to None.

  Returns:
      Tensor: dy/dx

  See `here <https://discuss.pytorch.org/t/how-to-compute-jacobian-matrix-in-pytorch/14968/14>`_
  """
  if grad_outputs is None:
    grad_outputs = torch.ones_like(y)
  grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True)[0]
  return grad


def get_value(x: torch.Tensor) -> np.ndarray:
  """
  An utility function to get values of the tensor either in the graph or on GPU.

  Args:
      x (torch.Tensor): Input tensor

  Returns:
      np.ndarray: A number version of the input tensor
  """
  return x.detach().cpu().numpy() 