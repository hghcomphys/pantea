from abc import ABCMeta, abstractmethod
from typing import Any


class StructureDataset(metaclass=ABCMeta):
    """
    A data container for atom data structure.

    Features:

    * it must access data item in a lazy mode.
    * it should be able to cache data via a `persist` input flag.
    """

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index) -> Any:
        pass
