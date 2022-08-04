# # TorchIP: Lennard-Jones potential
# An example notebook that shows how to reconstruct a Lennard-Jones potential using 
# high-dimensional neural network potential (HDNNP). 

# ### Imports

import sys
sys.path.append('../')

import torchip
from torchip.datasets import RunnerStructureDataset, ToStructure
from torchip.potentials import NeuralNetworkPotential

import torch
import logging
import pandas as pd
from pathlib import Path

# from dask.distributed import Client
# from torchip.config import TaskClient
# TaskClient.client = Client(memory_limit='4GB', n_workers=2, processes=True, threads_per_worker=2, dashboard_address=':8791')

# torchip.set_logging_level(logging.DEBUG)
torchip.manual_seed(2022)
torchip.device.DEVICE = torch.device("cpu")

potdir = Path('./LJ')

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