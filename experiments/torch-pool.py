import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# import torch.multiprocessing as mp
# from torch.multiprocessing import Pool, set_start_method

torch.set_num_threads(3)
print(torch.get_num_threads())
executor = ThreadPoolExecutor(max_workers=3)
# torch.set_num_threads(3)
# set_start_method('spawn', force=True)
# pool = Pool(3)

def kernel(x, y):
  # x, y = torch.rand(3)/(inp+1), torch.rand(3)/(inp+1)
  if x.max() > y.max():
    r = x 
  else:
    r = y
  for i in range(1, 1000000):
    r *= i
    r /= i
  return r

def show_result(future):
  print(future.result())

def main():
  futures = []
  # processes = []
  for i in range(3):
    print("run")
    # print(kernel(torch.rand(3), torch.rand(3)))
    future = executor.submit(kernel, torch.rand(3), torch.rand(3))
    future = torch.futures.Future()
    future.add_done_callback(show_result)
    futures.append( future )
    # p = mp.Process(target=kernel, args=(torch.rand(3), torch.rand(3)))
    # p.start()
    # processes.append(p)
    # pool.map(kernel, range(3))
  # for p in processes:
  #   p.join()


if __name__ == "__main__":
  main()
  # time.sleep(10)



