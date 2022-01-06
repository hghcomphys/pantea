
import numpy as np
# import torch

def find_neighbor(x, aid):
  result = np.where( np.linalg.norm(x-x[aid,:], axis=1) < R_CUT )[0]
  #result = np.linalg.norm(x-x[aid,:], axis=1) < R_CUT
  return result

def find_neighbors(x, nn, nb):
  for aid in range(N):
    neighbors_ = find_neighbor(x, aid)
    numOfNeighbors_ = len(neighbors_)
    nn[aid] = numOfNeighbors_
    nb[aid, 0:numOfNeighbors_] = neighbors_


if __name__ == "__main__":

  N = 1000
  DIM = 3
  R_CUT = 0.5
  np.random.seed(12345)

  x = np.random.rand(N, DIM)
  nn = np.zeros(N, dtype=np.int32)
  nb = np.zeros((N, N), dtype=np.int32)

  # print(x)
  print( find_neighbor(x, 0) )

  find_neighbors(x, nn, nb)
  print(nn)
  print(nb)