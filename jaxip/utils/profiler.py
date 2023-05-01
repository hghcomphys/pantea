from __future__ import annotations

import copy
import functools
import time
from collections import defaultdict
from typing import Any, Callable, Mapping

import pandas as pd

from jaxip.logger import logger

_default_profiler = None


class Profiler:
    """
    An implementation of a basic profiler for debugging.

    How to use it:
        1) add profile() as a decorator before each targeted method
        2) use context manager to collect stats from the region of interest
    """

    def __init__(self, name: str = "Profiler", sort_by: str = "cumtime") -> None:
        logger.debug(
            f"An instance of {self.__class__.__name__} object has been initialized"
        )
        self.name: str = name
        self.sort_by: str = sort_by
        self.active: bool = False
        self._walltimer: float = 0.0
        self.stats: Mapping[str, Any] = defaultdict(
            lambda: {
                "number_of_calls": 0,
                "cumulative_time": 0.0,
                "star_walltime": None,
                "end_walltime": 0.0,
            }
        )

    def __enter__(self) -> Profiler:
        """Start profiling."""
        logger.info(f"{self.name} started")
        self.active = True
        global _default_profiler
        _default_profiler = self
        self._walltimer = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback) -> None:
        """Stop profiling."""
        self.active = False
        global _default_profiler
        self.stats = copy.deepcopy(_default_profiler.stats)
        _default_profiler = None
        logger.info(f"{self.name} finished")
        logger.info(self)

    @staticmethod
    def profile(func) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            global _default_profiler
            if _default_profiler is not None and _default_profiler.active:
                start_time: float = time.perf_counter()
                retval = func(self, *args, **kwargs)
                end_time: float = time.perf_counter()
                func_info: str = f"{self.__class__.__name__}.{func.__name__}"
                _default_profiler.stats[func_info]["number_of_calls"] += 1
                _default_profiler.stats[func_info]["cumulative_time"] += (
                    end_time - start_time
                )
                if _default_profiler.stats[func_info]["start_walltime"] is None:
                    _default_profiler.stats[func_info]["start_walltime"] = (
                        time.perf_counter() - _default_profiler._walltimer
                    )
                _default_profiler.stats[func_info]["end_walltime"] = (
                    time.perf_counter() - _default_profiler._walltimer
                )
            else:
                retval = func(self, *args, **kwargs)
            return retval

        return wrapper

    def get_dataframe(self) -> pd.DataFrame:
        """Return a dataframe representation of profiler statistics."""
        if len(self.stats) == 0:
            pd.DataFrame({})

        df = defaultdict(list)
        for func, stats in self.stats.items():
            df["method"].append(func)
            for metric, value in stats.items():
                df[metric].append(value)
        return pd.DataFrame(df).sort_values(self.sort_by, ascending=False)

    def __repr__(self) -> str:
        df_: pd.DataFrame = self.get_dataframe()
        if df_ is not None:
            return f"Profiler statistics:\n{df_.to_string()}"


class Timer:
    """A class to measure elapsed time."""

    def __init__(self, name: str = "Timer") -> None:
        self.name: str = name
        self.elapsed_time: float = 0.0
        self._start: float

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback) -> None:
        self.elapsed_time = time.perf_counter() - self._start
        print(self)

    def __repr__(self) -> str:
        return f"{self.name} (elapsed time {self.elapsed_time:.8f} seconds)"


def timer(func: Callable):
    """A decorator to measure elapsed time when calling a function."""
    print(func.__name__)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(f"Timer {func.__name__} function"):
            retval = func(*args, **kwargs)
        return retval

    return wrapper
