"""

Ref: https://pytorch.org/tutorials/intermediate/dist_tuto.html
"""

#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def kernel(inp):
  x, y = torch.rand(3)/(inp+1), torch.rand(3)/(inp+1)
  if x.max() > y.max():
    r = x 
  else:
    r = y
  for i in range(1, 1000000):
    r *= i
    r /= i
  return r

def run(rank, size):
    """ Distributed function to be implemented later. """
    return kernel(rank)

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 3
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()