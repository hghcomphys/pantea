from typing import Protocol

from jaxip.atoms.structure import Structure


class DatasetInterface(Protocol):
    """
    A data container for atom data structure.

    Features:

    * it must access data item in a lazy mode.
    * it should be able to cache data via a `persist` input flag.
    """

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> Structure:
        ...
