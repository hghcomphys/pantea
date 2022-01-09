import torch

dtype = torch.double
device = torch.device("cpu")

class Box:
  """
  A class which contains cell info.
  Currently, it only works for orthogonal cell.
  """
  def __init__(self, cell):
    """
    Initialize box.
    """
    self.xlo = torch.tensor(0.0, dtype=dtype, device=device)
    self.ylo = torch.tensor(0.0, dtype=dtype, device=device)
    self.zlo = torch.tensor(0.0, dtype=dtype, device=device)
    self.xhi = cell[0, 0]
    self.yhi = cell[1, 1]
    self.zhi = cell[2, 2]