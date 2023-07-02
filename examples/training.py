# An example notebook for constructing a high-dimensional neural network potential (HDNNP).

import os

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import logging
from pathlib import Path

import jax.numpy as jnp

import jaxip
from jaxip.datasets import RunnerDataset
from jaxip.logger import set_logging_level
from jaxip.potentials import NeuralNetworkPotential
from jaxip.types import dtype as default_dtype

print(jaxip.__doc__)
print(f"(version {jaxip.__version__})\n")

set_logging_level(logging.INFO)
default_dtype.FLOATX = jnp.float64

potdir = Path("./LJ")

# Dataset
structures = RunnerDataset(Path(potdir, "input.data"), persist=True)
# structures = [structures[i] for i in range(len(structures))]

# Potential
nnp = NeuralNetworkPotential.from_file(Path(potdir, "input.nn"))

# nnp.load_scaler()
nnp.fit_scaler(structures)

# nnp.load_model()
history = nnp.fit_model(structures)
