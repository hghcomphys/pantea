import torch
import time
from lltm_cpp import kernel as cpp_kernel
from lltm_cpp import pkernel as cpp_pkernel

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, set_start_method

# torch.set_num_threads(3)
# print(torch.get_num_threads())
executor = ThreadPoolExecutor(max_workers=3)
# # torch.set_num_threads(3)
# # set_start_method('spawn', force=True)
# # pool = Pool(3)


# GPU version is too slow!
device = torch.device("cpu")


def run_kernel():
  """
  Regular python kernel
  """
  def kernel(z):
      tot = torch.sigmoid(z)
      for i in range(500000):
        tot += torch.sigmoid(z*i)
      return tot
  # Run kernel
  t0 = time.perf_counter()
  for i in range(3):
    ans = kernel(torch.rand(2, 10, device=device))
  print("Python kernel:", time.perf_counter() - t0)


def run_cpp_kernel():
  t0 = time.perf_counter()
  for i in range(3):
    ans = cpp_kernel(torch.rand(2, 10, device=device))
  print("CPP kernel:", time.perf_counter() - t0)


def run_cpp_pkernel():
  t0 = time.perf_counter()
  ans = cpp_pkernel() # parallel kernel
  print("CPP parallel kernel:", time.perf_counter() - t0)


def run_kernel_parallel():
  """
  multi-thread/process
  Future
  """
  def show_result(future):
    print(future.result())
  # Submit tasks
  t0 = time.perf_counter()
  futures = []
  processes = []
  for i in range(3):
    print("submit")
    # future = executor.submit(lltm_cpp.kernel, torch.rand(2, 10))
    # future = torch.futures.Future()
    # future.add_done_callback(show_result)
    # futures.append( future )
    p = mp.Process(target=cpp_kernel, args=(torch.rand(2, 10, device=device),))
    p.start()
    processes.append(p)
    # pool.map(kernel, range(3))
  for p in processes:
    p.join()
  print("CPP kernel:", time.perf_counter() - t0)


if __name__ == "__main__":
  run_kernel()
  run_cpp_kernel()
  run_cpp_pkernel()
  run_kernel_parallel()  # still GIL problem here!