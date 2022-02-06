import torch


def gradient(y, x, grad_outputs=None):
  """
  Compute dy/dx @ grad_outputs
  Ref: https://discuss.pytorch.org/t/how-to-compute-jacobian-matrix-in-pytorch/14968/14
  """
  if grad_outputs is None:
    grad_outputs = torch.ones_like(y)
  grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True)[0]
  return grad
