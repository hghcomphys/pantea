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
    count = 0
    # Loop over structure data
    for index, data in enumerate(structure_loader.get_data(), start=1):
        count += 1
        if (between is not None) and ( (index < between[0]) or (index > between[1]) ):
            continue
        logger.info(f"Reading structure #{index}")
        structures.append( Structure(data) )   
    logger.info(f"Read {len(structures)} of {count} structures")
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
