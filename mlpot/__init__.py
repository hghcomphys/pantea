"""
MLPOT - A Framework for Developing Machine Learning Interatomic Potential
=========================================================================

MLPOT is a software _framework_ written in Python to facilitate the development of emerging 
machine learning interatomic potentials. It is intended to help researchers to quickly construct 
their ML-based potentials and allowing to perform large-scale molecular dynamics simulations 
of complex materials in computational physics and chemistry.

The core implementation of MLPOT is based on the PyTorch package, and its C++ API, which 
provides two high-level features including optimized tensor computation and automatic 
differentiation.
"""

from .logger import *
from .config import *


__version__ = "0.3.0"

logger.debug(f"{__doc__}")
logger.debug(f"Version: {__version__}")
