
import numpy as np
# import torch

N = 10000
DIM = 3
R_CUT = 0.5
np.random.seed(12345)

x = np.random.rand(N, DIM)
nn = np.zeros(N, dtype=np.int32)
nb = np.zeros((N, N), dtype=np.int32)

def find_neighbor(aid):
  result = np.where( np.linalg.norm(x-x[aid,:], axis=1) < R_CUT )
  #result = np.linalg.norm(x-x[aid,:], axis=1) < R_CUT
  return result

def find_neighbors():
  for aid in range(N):
    neighbors_ = find_neighbor(aid)[0]
    numOfNeighbors_ = len(neighbors_)
    nn[aid] = numOfNeighbors_
    nb[aid, 0:numOfNeighbors_] = neighbors_

# print(x)
print( find_neighbor(0) )

find_neighbors()
print( nn )
print( nb )