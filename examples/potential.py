# An example notebook for constructing a high-dimensional neural network potential (HDNNP).

import os

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import logging
from pathlib import Path

import jax.numpy as jnp
#import pandas as pd

from jaxip.datasets import RunnerStructureDataset
from jaxip.logger import set_logging_level
from jaxip.potentials import NeuralNetworkPotential
from jaxip.types import dtype as default_dtype

# default_dtype.FLOATX = jnp.float64
# set_logging_level(logging.INFO)

potdir = Path("./LJ")

# Dataset
structures = RunnerStructureDataset(Path(potdir, "input.data"), persist=True)
structures = [structures[i] for i in range(len(structures))]

# Potential
nnp = NeuralNetworkPotential.create_from(Path(potdir, "input.nn"))

# nnp.load_scaler()
nnp.fit_scaler(structures)

# nnp.load_model()
history = nnp.fit_model(structures)

# df = pd.DataFrame(history)
# print(df.tail())
