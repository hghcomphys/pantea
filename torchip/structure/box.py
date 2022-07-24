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
  def __init__(self, 
      lattice,
      dtype: torch.dtype = None,
      device: torch.device = None,
    ) -> None:
    """
    Initialize the simulation box (super-cell).

    :param lattice: Lattice matrix (3x3 array)
    :param dtype: Data type of internal tensors which represent structure, defaults to None
    :type dtype: torch.dtype, optional
    :param device: Device on which tensors are allocated, defaults to None
    :type device: torch.device, optional
    """    

    self.lattice = torch.tensor(lattice, dtype=dtype, device=device)

    # Check lattice matrix shape
    if self.lattice.shape != (3, 3):
      logger.error(f"Unexpected lattice dimension {self.lattice.shape}", exception=ValueError)

    self.xlo = torch.tensor(0.0, dtype=dtype, device=device)
    self.ylo = torch.tensor(0.0, dtype=dtype, device=device)
    self.zlo = torch.tensor(0.0, dtype=dtype, device=device)
    self.xhi = self.lattice[0][0]
    self.yhi = self.lattice[1][1]
    self.zhi = self.lattice[2][2]

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
