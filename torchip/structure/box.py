from ..logger import logger
from ..config import dtype as _dtype
from ..config import device as _device
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
  def __init__(self, lattice, dtype=None, device=None) -> None:
    """
    Initialize simulation box (super-cell).

    :param lattice: Lattice matrix (3x3 array)
    :param dtype: Data type of internal tensors which represent structure, defaults to None
    :type dtype: torch.dtype, optional
    :param device: Device on which tensors are allocated, defaults to None
    :type device: torch.device, optional
    """    
    self.dtype = dtype if dtype else _dtype.FLOAT
    self.device = device if device else _device.DEVICE

    # Create lattice tensor
    self.lattice = torch.tensor(lattice, dtype=self.dtype, device=self.device)

    if self.lattice.shape != (3, 3):
      logger.error(f"Unexpected lattice dimension {self.lattice.shape}", exception=ValueError)

  @staticmethod
  def _apply_pbc(dx: Tensor, lattice: Tensor) -> Tensor:
    """
    [Kernel]
    Apply the periodic boundary condition (PBC) along x,y, and z directions.

    :param dx: Position difference
    :type dx: Tensor
    :param lattice: lattice matrix
    :type lattice: Tensor
    :return: PBC applied position
    :rtype: Tensor
    """    
    # TODO: use broadcasting
    for i in range(3):
      l = lattice[i, i]
      dx_i = dx[..., i]; dx[..., i] = torch.where( dx_i >  0.5E0*l, dx_i - l, dx_i)
      dx_i = dx[..., i]; dx[..., i] = torch.where( dx_i < -0.5E0*l, dx_i + l, dx_i)
    return dx

  def apply_pbc(self, dx: Tensor) -> Tensor:
    """
    Apply the periodic boundary condition (PBC) on input tensor.

    :param dx: Position difference
    :type dx: Tensor
    :return: PBC applied position difference
    :rtype: Tensor
    """    
    return Box._apply_pbc(dx, self.lattice)

  @property
  def xlo(self) -> Tensor:
    return torch.tensor(0.0, dtype=self.dtype, device=self.device)

  @property
  def ylo(self) -> Tensor:
    return torch.tensor(0.0, dtype=self.dtype, device=self.device)

  @property
  def zlo(self) -> Tensor:
    return torch.tensor(0.0, dtype=self.dtype, device=self.device)

  @property
  def xhi(self) -> Tensor:
    return self.lattice[0, 0]

  @property
  def yhi(self) -> Tensor:
    return self.lattice[1, 1]

  @property
  def zhi(self) -> Tensor:
    return self.lattice[2, 2]

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
