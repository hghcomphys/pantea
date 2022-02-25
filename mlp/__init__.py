"""Machine Learning Potential (MLP) framework is a generic and GPU-accelerated package 
written in Python/C++ to facilitate the development of emerging machine learning 
interatomic potentials. Such potentials are employed to perform large-scale molecular 
dynamics simulations of complex materials in computational physics and chemistry."""

from .logger import logger
from .config import CFG


__version__ = "0.0.1"

# Version
logger.info(f"{__doc__}")
logger.info(f"MLP framework version: {__version__}")

# CUDA availability
logger.info(f"CUDA availability: {CFG['is_cuda']}")
logger.info(f"Default device: '{CFG['device']}'")