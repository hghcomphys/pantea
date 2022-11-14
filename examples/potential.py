# # LMPOT: Lennard-Jones potential
# An example notebook that shows how to reconstruct a Lennard-Jones potential using
# high-dimensional neural network potential (HDNNP).

# ### Imports

import sys

sys.path.append("../")

import mlpot
from mlpot.datasets import RunnerStructureDataset
from mlpot.potentials import NeuralNetworkPotential
import torch
import pandas as pd
from pathlib import Path

# mlpot.set_logging_level(logging.DEBUG)
mlpot.manual_seed(2022)
mlpot.device.DEVICE = torch.device("cpu")

potdir = Path("./LJ")

# Dataset
structures = RunnerStructureDataset(Path(potdir, "input.data"), persist=True)

# Potential
nnp = NeuralNetworkPotential(Path(potdir, "input.nn"))

# nnp.load_scaler()
nnp.fit_scaler(structures)

# nnp.load_model()
history = nnp.fit_model(structures, epochs=1, validation_split=0.20)

# df = pd.DataFrame(history)
# print(df.tail())