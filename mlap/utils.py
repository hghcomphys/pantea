from .logger import logger
from .structure import Structure
from .loader import StructureLoader
from typing import List
import torch


def read_structures(structure_loader: StructureLoader) -> List[Structure]:
    """
    Read the input structure loader and return a list of structures.
    """
    structures = []
    # Loop over structure data
    for index, data in enumerate(structure_loader.get_data(), start=1):
        logger.info(f"Reading structure {index}")
        structures.append( Structure(data) )   
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
