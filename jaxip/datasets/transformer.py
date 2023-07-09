from typing import Any, Dict, Optional, Protocol

from jaxip.atoms.structure import Structure
from jaxip.types import Dtype, _dtype


class TransformerInterface(Protocol):
    """
    A base transformer class which applies on the structure dataset.
    """

    def __call__(self, data: Dict, **kwargs: Any) -> Structure:
        ...

    def __repr__(self) -> str:
        ...


class ToStructure(TransformerInterface):
    """Transform a dictionary of data into to a Structure."""

    def __init__(
        self,
        r_cutoff: Optional[float] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        self.r_cutoff: Optional[float] = r_cutoff
        self.dtype: Dtype = _dtype.FLOATX if dtype is None else dtype

    def __call__(self, data: Dict[str, Any]) -> Structure:
        return Structure.from_dict(data, dtype=self.dtype)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
