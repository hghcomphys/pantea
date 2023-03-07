from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any

from jaxip.structure.structure import Structure


class StructureDataset(metaclass=ABCMeta):
    """
    A data container for atom data structure.

    Features:

    * it must access data item in a lazy mode.
    * it should be able to cache data via a `persist` input flag.
    """

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, index) -> Any:
        ...
    
    @abstractmethod
    def __next__(self) -> Structure:
        ...

    @abstractmethod
    def __iter__(self) -> StructureDataset:
        ...
