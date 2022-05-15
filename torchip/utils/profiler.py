from __future__ import annotations
from ..logger import logger
from collections import defaultdict
import functools
import time
import copy


_default_profiler = None

class Profiler:
  # """
  # An implementation of a basic profiler for investigating code performace.
  # How to use it:
  #     1. add profile() as a decorator before each targeted method 
  #     2. use context manager to collect stattistics from the region of interest
  #         Or otherwise those descorators are ignored - there will be small overheads
  # TODO: testing multi-thread applications
  # TODO: no support for multi-process 
  # """
  
  def __init__(self, name: str = 'Profiler', sort_by: str = 'cumtime'):
    logger.debug(f"An instance of {self.__class__.__name__} object has been initialized")
    self.name = name
    self.sort_by= sort_by
    self.active = None
    self._walltimer = None
    self.stats = defaultdict( lambda: {
                        'ncalls': 0,      # number of calls
                        'cumtime': 0.0,   # cumulative time
                        'startwt': None,  # start walltime
                        'endwt': 0.0,     # end walltime
                        })
  
  def __enter__(self) -> Profiler:
    """
    Enter the profiler with context.
    """
    logger.info(f"{self.name} started")
    self.active = True
    global _default_profiler
    _default_profiler = self
    self._walltimer = time.perf_counter()
    return self

  def __exit__(self, type, value, traceback) -> None:
    """
    Exit the with context.
    """
    self.active = False
    global _default_profiler
    self.stats = copy.deepcopy(_default_profiler.stats)
    _default_profiler = None
    logger.info(f"{self.name} finished")
    logger.info(self)
    
  @staticmethod
  def profile(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):      
      global _default_profiler
      if _default_profiler is not None and _default_profiler.active:
        start_time = time.perf_counter()    
        retval = func(self, *args, **kwargs)
        end_time = time.perf_counter()       
        func_info = f"{self.__class__.__name__}.{func.__name__}" 
        _default_profiler.stats[func_info]['ncalls'] += 1
        _default_profiler.stats[func_info]['cumtime'] += end_time - start_time
        if _default_profiler.stats[func_info]['startwt'] is None:
          _default_profiler.stats[func_info]['startwt'] = time.perf_counter() - _default_profiler._walltimer
        _default_profiler.stats[func_info]['endwt'] = time.perf_counter() - _default_profiler._walltimer
      else:
        retval = func(self, *args, **kwargs)
      return retval
    return wrapper

  def get_dataframe(self): # -> pd.DataFrame:
      """
      Return a dataframe representation of profiler statistics.
      """
      if len(self.stats) == 0:
        return None
      import pandas as pd
      df = defaultdict(list)
      for func, stats in self.stats.items():
        df['method'].append(func)
        for metric, value in stats.items():
          df[metric].append(value)
      return pd.DataFrame(df).sort_values(self.sort_by, ascending=False)

  def __repr__(self) -> str:
    df_ = self.get_dataframe()
    if df_ is not None:
      return f"Profiler statistics (dataframe):\n{df_.to_string()}"


class Timer():
  """
  A class to measure elapsed time.
  """
  def __init__(self, name: str = "Timer"):
    self.name = name
    self.elapsed_time = None

  def __enter__(self):
    self._start = time.perf_counter()
    return self

  def __exit__(self, type, value, traceback):
    self.elapsed_time = time.perf_counter() - self._start
    logger.info(self)

  def __repr__(self) -> str:
      return f"{self.name} (elapsed time {self.elapsed_time:.8f} seconds)"
  

def timer(func):
  """
  A decorator to measure elapsed time when calling a function.
  """
  print(func.__name__)
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    with Timer(f"Timer {func.__name__} function"):
      retval = func(*args, **kwargs)
    return retval
  return wrapper