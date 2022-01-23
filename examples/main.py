import sys
sys.path.append('../')

from mlap.config import CFG
from mlap.logger import logger
from mlap.loaders import RunnerStructureLoader as StructureLoader
from mlap.descriptors import ASF, CutoffFunction, G1, G2
from mlap.potentials import NeuralNetworkPotential
from mlap import gradient, read_structures
# from mlap.models import NeuralNetwork
import torch

torch.manual_seed(2022)

# TODO: move to root mlap import
logger.info(f"CUDA availability: {CFG['is_cuda']}")
logger.info(f"Default device: '{CFG['device']}'")

CFG.set("device", "cpu")
# print(CFG["device"])

# Read structure
loader = StructureLoader("/home/hossein/Desktop/n2p2_conf_3816/input.data")
structures = read_structures(loader, between=(1, 10)) # TODO: structure index 0 or 1?
str0 = structures[0] 
print(str0.lattice)
print(str0.atype)
print(str0.box.length)
print(str0.calculate_distance(aid=0, detach=True))
# Neighbor list
str0.update_neighbor()
print(str0.neighbor_number)
print(str0.neighbor_index)

# Cutoff functions
# cfn = CutoffFunction(0.5, "tanh")
# r = torch.rand(10, requires_grad=True)
# val = cfn(r)
# print(val)
# print(gradient(val, r))

# Define descriptor
dsr = ASF(element="H")
r_cutoff, cutoff_type = 12.0, "tanh"
dsr.add(G1(r_cutoff, cutoff_type), "H")
dsr.add(G2(r_cutoff, cutoff_type, r_shift=0.0, eta=0.3), "H")
dsr.add(G2(r_cutoff, cutoff_type, r_shift=0.0, eta=0.3), "O")
dsr.add(G2(r_cutoff, cutoff_type, r_shift=0.0, eta=0.5), "H")
# Calculate values
str1 = structures[1]
val = dsr(str1, aid=1)
print(val)
print(gradient(val, str1.position)[:10]) 

# # Potential
# pot = NeuralNetworkPotential("input.nn")
# # res = pot.train(loader)
# # print(res)







