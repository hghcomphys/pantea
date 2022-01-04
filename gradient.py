import numpy as np
from numpy.testing._private.utils import requires_memory
import torch

N = 10
DIM = 3
R_CUT = 0.5
np.random.seed(12345)

def kernel(x: torch.tensor) -> torch.tensor:
  return x*x + x[0, 0]*x[0, 1]*x[0, 2]

# https://discuss.pytorch.org/t/how-to-compute-jacobian-matrix-in-pytorch/14968/14
def gradient(y, x, grad_outputs=None):
    """
    Compute dy/dx @ grad_outputs
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True)[0]
    return grad

def jacobian(y, x): # This might be slow use torch.autograd.functional.jacobian instead!
    """
    Compute dy/dx = dy/dx @ grad_outputs; 
    for grad_outputs in [1, 0, ..., 0], [0, 1, 0, ..., 0], ...., [0, ..., 0, 1]
    """
    jac = torch.zeros(y.shape[0], x.shape[0]) 
    for i in range(y.shape[0]):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[i] = 1
        jac[i] = gradient(y, x, grad_outputs = grad_outputs)
    return jac

def divergence(y, x): # This one also might be slow!
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


if __name__ == "__main__":
  x = torch.tensor( [[1, 2, 3] for _ in range(N)], dtype=torch.float, requires_grad=True)
  print(x)

  y = kernel(x)
  grad = gradient(y, x)
  print(grad)

  # job = torch.autograd.functional.jacobian(kernel, x)
  # print(job.shape)