"""
TorchIP is a generic and GPU-accelerated framework written in Python/C++ to facilitate 
the development of emerging machine learning interatomic potentials. TorchIP is intended 
to help researchers to construct potentials that are employed to perform large-scale 
molecular dynamics simulations of complex materials in computational physics and chemistry.

The core implementation of TorchIP is based on the PyTorch package and its C++ API which 
provides two high-level features including optimized tensor computation and automatic 
differentiation
"""

from .logger import logger
from .config import *


__version__ = "0.4.1"

logger.debug(f"{__doc__}")
logger.debug(f"TorchIP {__version__}")
