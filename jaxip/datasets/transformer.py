from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional

from jaxip.structure.structure import Structure
from jaxip.types import Dtype
from jaxip.types import dtype as _dtype


class Transformer(metaclass=ABCMeta):
    """
    A base transformer class which applies on the structure dataset.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        pass


class ToStructure(Transformer):
    """Transform a dictionary of data into to a Structure."""

    def __init__(
        self,
        r_cutoff: Optional[float] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        self.r_cutoff: Optional[float] = r_cutoff
        self.dtype: Dtype = dtype if dtype is not None else _dtype.FLOATX

    def __call__(self, data: Dict[str, Any]) -> Structure:  # type: ignore
        return Structure.create_from_dict(
            data, r_cutoff=self.r_cutoff, dtype=self.dtype
        )
