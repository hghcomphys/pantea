from .logger import logger
from .structure import Structure
from .loaders import StructureLoader
from typing import List, Tuple
import torch


def read_structures(structure_loader: StructureLoader, between: Tuple[int, int]=None) -> List[Structure]:
  """
  Read the input structure loader and return a list of structures.
  """
  structures = []
  index = 0
  # Loop over structure data
  structure_generator = structure_loader.get_data()
  while True:
    index += 1
    try:
      if (between is not None) and ( (index < between[0]) or (index > between[1]) ):
        structure_loader.ignore_next()
        next(structure_generator)
      else:
        logger.debug(f"Reading structure #{index}")
        data = next(structure_generator)
        structures.append( Structure(data) )   
    except StopIteration:
      index -= 1 # correct over counting
      break

  logger.info(f"Read {len(structures)} of {index} structures: between={between if between is not None else 'all'}")
  return structures  

def gradient(y, x, grad_outputs=None):
  """
  Compute dy/dx @ grad_outputs
  Ref: https://discuss.pytorch.org/t/how-to-compute-jacobian-matrix-in-pytorch/14968/14
  """
  if grad_outputs is None:
    grad_outputs = torch.ones_like(y)
  grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True)[0]
  return grad
