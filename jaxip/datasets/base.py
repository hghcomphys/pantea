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

    def __repr__(self) -> str:
        return "{C}({attrs})".format(  # @{id:x}
            C=self.__class__.__name__,
            # id=id(self) & 0xFFFFFF,
            attrs=", ".join(
                "{}={!r}".format(k, v)
                for k, v in self.__dict__.items()
                if not k.startswith("_")
            ),
        )
