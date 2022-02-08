import sys
sys.path.append('../')

from mlp.config import CFG
from mlp.logger import logger
from mlp.loaders import RunnerStructureLoader as StructureLoader
from mlp.loaders import read_structures
from mlp.descriptors import ASF, CutoffFunction, G1, G2, G3
from mlp.potentials import NeuralNetworkPotential
from mlp.utils import gradient
# from mlap.models import NeuralNetwork

import torch
torch.manual_seed(2022)

# TODO: move to root mlap import
logger.info(f"CUDA availability: {CFG['is_cuda']}")
logger.info(f"Default device: '{CFG['device']}'")

# torch.set_num_threads(2)
# CFG.set("device", "cpu")
# print(CFG["device"])

# Read structure
path = '/home/hossein/n2p2/examples/nnp-scaling/H2O_RPBE-D3/{}' #"/home/hossein/Desktop/n2p2_conf_3816/{}"
loader = StructureLoader(path.format("input.data")) 
# structures = read_structures(loader, between=(1, 5)) # TODO: structure index 0 or 1?
# str0 = structures[0] 
# print(str0.lattice)
# print(str0.atype)
# print(str0.box.length)
# print(str0.calculate_distance(aid=0, detach=True))
# # Neighbor list
# str0.update_neighbor()
# print(str0.neighbor.number)
# print(str0.neighbor.index)
# print(str0.select("O"))

# Cutoff functions
# cfn = CutoffFunction(0.5, "tanh")
# r = torch.rand(10, requires_grad=True)
# val = cfn(r)
# print(val)
# print(gradient(val, r))

# Define descriptor
# dsr = ASF(element="H")
# r_cutoff, cutoff_type = 12.0, "tanh"
# dsr.add(G1(r_cutoff, cutoff_type), "H")
# dsr.add(G2(r_cutoff, cutoff_type, r_shift=0.0, eta=0.3), "H")
# dsr.add(G2(r_cutoff, cutoff_type, r_shift=0.0, eta=0.3), "O")
# dsr.add(G2(r_cutoff, cutoff_type, r_shift=0.0, eta=0.5), "H")
# dsr.add( G3(r_cutoff, cutoff_type, eta=0.0010, zeta=2.0, lambda0=1.0, r_shift=12.0), "H", "H")
# # Calculate values
# val = dsr(str0, aid=range(100))
# print(val)
# print(gradient(val[0], str0.position)[:10]) 

# Potential
pot = NeuralNetworkPotential(path.format("input.nn"))
pot.fit_scaler(loader)
# pot.read_scaler("scaling.data")

str0 = read_structures(loader, between=(1, 1))[0]
for element in pot.descriptor.keys():
  print("Element:", element)
  print(pot.scaler[element].__dict__)
  val = pot.descriptor[element](str0, str0.select(element)[:1])
  print(val)
  print(gradient(val, str0.position)[:10])



 






