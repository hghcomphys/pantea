"""
Pantea is an optimized Python library on basis of Google JAX that enables development of machine
learning inter-atomic potentials for use in computational physics, chemistry, and material science.
These potentials are necessary for conducting large-scale molecular dynamics simulations of complex
materials with ab initio accuracy.
"""

import os

from pantea._version import __version__ as version

__author__ = """Hossein Ghorbanfekr"""
__email__ = "hgh.comphys@gmail.com"
__version__ = version


os.environ["JAX_ENABLE_X64"] = "1"  # enable double precision
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # disable memory preallocation
