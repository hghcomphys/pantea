from typing import Dict

from mlpot.base import _Base
from mlpot.structure.structure import Structure


class Transformer(_Base):
    """
    A base transformer class which applies on the structure dataset.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ToStructure(Transformer):
    """
    An utility transformer that converts a structure dataset (dictionary) into to a **Structure** instance.
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def __call__(self, data: Dict) -> Structure:
        return Structure(data, **self.kwargs)
