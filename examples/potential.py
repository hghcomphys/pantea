# An example notebook that shows how to construct a potential using
# high-dimensional neural network potential (HDNNP).

import os

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import logging
from pathlib import Path

import jax.numpy as jnp
import pandas as pd

from mlpot.datasets import RunnerStructureDataset
from mlpot.logger import set_logging_level
from mlpot.potentials import NeuralNetworkPotential
from mlpot.types import dtype as default_dtype

# default_dtype.FLOATX = jnp.float64

# set_logging_level(logging.INFO)

potdir = Path("./H2O_2")

# Dataset
structures = RunnerStructureDataset(Path(potdir, "input.data"), persist=True)
structures = [structures[i] for i in range(10)]

# Potential
nnp = NeuralNetworkPotential(Path(potdir, "input.nn"))

# nnp.load_scaler()
nnp.fit_scaler(structures)

# nnp.load_model()
# history = nnp.fit_model(structures, epochs=10)

# df = pd.DataFrame(history)
# print(df.tail())
