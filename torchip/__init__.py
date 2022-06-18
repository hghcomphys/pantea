"""
.. image:: ./images/logo.png
  :width: 300

TorchIP is a generic and GPU-accelerated framework written in Python/C++ to facilitate 
the development of emerging machine learning interatomic potentials. TorchIP is intended 
to help researchers to construct potentials that are employed to perform large-scale 
molecular dynamics simulations of complex materials in computational physics and chemistry.

The core implementation of TorchIP is based on the PyTorch package and its C++ API which 
provides two high-level features including optimized tensor computation and automatic 
differentiation
"""

from .logger import logger
from .config import CFG


__version__ = "0.4.0"

# Version
logger.debug(f"{__doc__}")
logger.info(f"TorchIP {__version__}")
logger.info(f"CUDA availability: {CFG['is_cuda']}")
logger.info(f"Default device: '{CFG['device']}'")
logger.info(f"Default precision: '{CFG['dtype']}'")
