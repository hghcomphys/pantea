"""
MLPOT - A machine-learning framework for development of interatomic potential
=============================================================================

MLPOT is a Python package to facilitate development of the emerging machine-learning (ML)
interatomic potentials in computational physics and chemistry. Such potentials are essential
for performing large-scale molecular dynamics (MD) simulations of complex materials at the
atomic scale and with ab initio accuracy.
"""

from .logger import *
from .config import *


__version__ = "0.3.0"

logger.debug(f"{__doc__}")
logger.debug(f"Version: {__version__}")
