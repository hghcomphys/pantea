"""
TorchIP - PyTorch Interatomic Potential
=======================================

TorchIP is a generic and GPU-accelerated software framework written in Python/C++ to facilitate 
the development of emerging machine learning interatomic potentials. TorchIP is intended to help 
researchers to construct ML-based potentials that are employed to perform large-scale molecular 
dynamics simulations of complex materials in computational physics and chemistry.

The core implementation of TorchIP is based on the PyTorch package, and its C++ API, which 
provides two high-level features including optimized tensor computation and automatic 
differentiation.
"""

from .logger import *
from .config import *



__version__ = "0.6.0"

logger.debug(f"{__doc__}")
logger.debug(f"Version: {__version__}")
