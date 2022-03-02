import torch

@torch.jit.script 
def kernel(x, y):
  # x, y = torch.rand(3)/(inp+1), torch.rand(3)/(inp+1)
  if x.max() > y.max():
    r = x 
  else:
    r = y
  for i in range(1, 200000):
    r *= i
    r /= i
  return r

# def show_result(future):
#   print(future.value())

# @torch.jit.script 
# def main():
#   # futures = []
#   # processes = []
#   for i in range(3):
#     print("run")
#     # print(kernel(torch.rand(3), torch.rand(3)))
#     future = torch.jit._fork(kernel, torch.rand(3), torch.rand(3))
#     # future.add_done_callback(show_result)
#     # futures.append( future )

#   # tot = torch.rand(3)
#   # for future in futures:
#   #   print(future.result())
#   #   result = torch.jit._wait(future)
#   #   print(result)
#   #   tot += result
#   # print(tot)


# if __name__ == "__main__":
#   main()
#   # time.sleep(10)



import torch
from torch import Tensor
def foo(a : Tensor, b : Tensor) -> Tensor:
    return kernel(a, b) #a + b

def bar(a):
  fut1 : torch.jit.Future[Tensor] = torch.jit.fork(foo, a, b=torch.rand(3))
  fut2 : torch.jit.Future[Tensor] = torch.jit.fork(foo, a, b=torch.rand(3))
  return torch.jit.wait(fut1), torch.jit.wait(fut2)

script_bar = torch.jit.script(bar)
input = torch.rand(3)
# only the scripted version executes asynchronously
# assert script_bar(input) == bar(input)
# trace is not run asynchronously, but fork is captured in IR
graph = torch.jit.trace(bar, (input,)).graph
# assert "fork" in str(graph)
print(graph)
print(script_bar(input))