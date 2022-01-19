from .logger import logger
import torch

dtype = torch.double
device = torch.device("cpu")

class Box:
  """
  A class which contains lattice info.
  Currently, it only works for orthogonal lattice.
  """
  def __init__(self, lattice: torch.Tensor) -> None:
    """
    Initialize box.
    """
    # Check lattice matrix shape
    if lattice.shape != (3, 3):
      msg = f"Unexpected lattice dimension {lattice.shape}"
      logger.error(msg)
      raise ValueError(msg)

    self.xlo = torch.tensor(0.0, dtype=dtype, device=device)
    self.ylo = torch.tensor(0.0, dtype=dtype, device=device)
    self.zlo = torch.tensor(0.0, dtype=dtype, device=device)
    self.xhi = lattice[0, 0]
    self.yhi = lattice[1, 1]
    self.zhi = lattice[2, 2]