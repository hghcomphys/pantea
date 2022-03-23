import numpy as np
from numpy.core.numeric import zeros_like
from numpy.testing._private.utils import requires_memory
import torch


def kernel(x: torch.tensor, aid) -> torch.tensor:
  y = torch.tensor(0.0, dtype=torch.float)
  dis = torch.norm(x-x[0], dim=1)
  neighbors_ = torch.nonzero( dis < R_CUT, as_tuple=True)[0]
  for nb in neighbors_.detach().tolist():
      print(nb)
      y += torch.exp(-dis[nb]**2)
  return y

def gradient(y, x, grad_outputs=None):
    """
    Compute dy/dx @ grad_outputs
    Ref: https://discuss.pytorch.org/t/how-to-compute-jacobian-matrix-in-pytorch/14968/14
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True)[0]
    return grad


if __name__ == "__main__":

  N = 10
  DIM = 3
  R_CUT = 0.5
  torch.manual_seed(2022)

  x = torch.rand( N, DIM, dtype=torch.float, requires_grad=True)
  print(x)

  # neighbors_ = torch.nonzero( torch.norm(x-x[0], dim=1) < R_CUT, as_tuple=True)[0]
  # for nb in neighbors_.detach().tolist():
  #   print(nb)

  y = kernel(x, 0)
  print(y)
  grad = gradient(y, x)
  print(grad)
