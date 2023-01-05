from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional

from jaxip.base import _Base
from jaxip.structure.structure import Structure
from jaxip.types import Dtype
from jaxip.types import dtype as _dtype


class Transformer(_Base, metaclass=ABCMeta):
    """
    A base transformer class which applies on the structure dataset.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        pass


class ToStructure(Transformer):
    """Transform a dictionary of data into to a `Structure`."""

    def __init__(
        self,
        r_cutoff: Optional[float] = None,
        dtype: Dtype = _dtype.FLOATX,
    ) -> None:
        self.r_cutoff: Optional[float] = r_cutoff
        self.dtype: Dtype = dtype

    def __call__(self, data: Dict[str, Any]) -> Structure:  # type: ignore
        return Structure.create_from_dict(
            data, r_cutoff=self.r_cutoff, dtype=self.dtype
        )
