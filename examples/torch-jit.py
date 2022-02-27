import torch
import time

@torch.jit.script 
def kernel(x, y):
  if x.max() > y.max():
    r = x 
  else:
    r = y
  for i in range(1, 1000000):
    r *= i
    r /= i
  return r

# @torch.jit.script 
def main():
  futures = []
  for i in range(3):
    print("fork")
    future = torch.jit._fork(kernel, torch.rand(3), torch.rand(3))
    #futures.append( future )

  # tot = torch.rand(3)
  # for future in futures:
  #   result = torch.jit._wait(future)
  #   print(result)
  #   tot += result
  # print(tot)


if __name__ == "__main__":
  main()
  # time.sleep(10)