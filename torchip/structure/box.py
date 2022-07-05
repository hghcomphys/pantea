from keyword import kwlist
from ..logger import logger
from ..config import dtype, device
from typing import Tuple
from torch import Tensor
import torch


class Box:
  """
  Box class extract Box info from the lattice matrix.
  Currently, it only works for orthogonal lattice.
  TODO: box variables as numpy or pytorch?
  TODO: triclinic lattice
  """
  def __init__(self, lattice: Tensor) -> None:
    
    # Check lattice matrix shape
    if lattice.shape != (3, 3):
      logger.error(f"Unexpected lattice dimension {lattice.shape}", exception=ValueError)

    self.lattice = lattice
    kwargs = { "dtype": dtype.FLOATX, "device": device.DEVICE }
    self.xlo = torch.tensor(0.0, **kwargs)
    self.ylo = torch.tensor(0.0, **kwargs)
    self.zlo = torch.tensor(0.0, **kwargs)
    self.xhi = lattice[0, 0]
    self.yhi = lattice[1, 1]
    self.zhi = lattice[2, 2]

  @property
  def lx(self) -> Tensor:
    return self.xhi - self.xlo

  @property
  def ly(self) -> Tensor:
    return self.yhi - self.ylo

  @property
  def lz(self) -> Tensor:
    return self.zhi - self.zlo

  @property
  def length(self) -> Tuple[Tensor]:
    return self.lx, self.ly, self.lz
