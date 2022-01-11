from .structure import Structure
import torch

dtype = torch.double
device = torch.device("cpu")


def gradient(y, x, grad_outputs=None):
    """
    Compute dy/dx @ grad_outputs
    Ref: https://discuss.pytorch.org/t/how-to-compute-jacobian-matrix-in-pytorch/14968/14
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True)[0]
    return grad
    

class ASF:
  """
  Atomic Symmetry Function.
  Base class (Descriptor) -> Derived class (ASF)
  ASF is vector of different radial and angular terms.
  """

  def __init__(self):
    pass

  def __call__(self, structure:Structure, aid: int) -> torch.tensor: 
    """
    Calculate descriptor values for the input given structure.
    """
    x = structure.pos
    nn  = structure._neighbor.nn
    ngb = structure._neighbor.ngb
    self.r_cutoff = structure._neighbor.r_cutoff  # has to be set durning class instantiation

    result = torch.tensor(0.0, dtype=torch.float)
    rij = torch.norm(x[ngb[aid, :nn[aid]]]-x[0], dim=1)
    neighbors_ = torch.nonzero( rij < self.r_cutoff, as_tuple=True)[0]
    for nb in neighbors_.detach().tolist():
        result = result + self.kernel(rij[nb])
    return result

  def kernel(self, rij: torch.tensor) -> torch.tensor:
    # TODO: define cutoff function class
    # TODO: improve design for kernel arguments
    return torch.exp(-rij**2) * torch.tanh(1.0 - rij/self.r_cutoff).pow(3)



