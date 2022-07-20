# # TorchIP: Lennard-Jones potential
# An example notebook that shows how to reconstruct a Lennard-Jones potential using 
# high-dimensional neural network potential (HDNNP). 

# ### Imports

import sys
sys.path.append('../')

import torchip as tp
from torchip.datasets import RunnerStructureDataset, ToStructure
from torchip.potentials import NeuralNetworkPotential

import torch
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(2022)
torch.manual_seed(2022);

# logger.set_level(logging.DEBUG)

# Set device eigher cpu or cuda (gpu)
tp.device.DEVICE = "cpu"

potdir = Path('./H2O')


# Dataset
structures = RunnerStructureDataset(Path(potdir, "input.data"), transform=ToStructure(), persist=False)

# Potential
pot = NeuralNetworkPotential(Path(potdir, "input.nn"))

# pot.load_scaler()
pot.fit_scaler(structures)


# # ### Model

# # #### Training

# # pot.load_model()
# history = pot.fit_model(structures, epochs=5, validation_split=0.20)

# df = pd.DataFrame(history)
# print(df.tail())
