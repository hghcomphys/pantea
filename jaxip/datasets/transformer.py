from typing import Any, Dict, Optional, Protocol

from jaxip.atoms.structure import Structure
from jaxip.types import Dtype
from jaxip.types import dtype as _dtype


class Transformer(Protocol):
    """
    A base transformer class which applies on the structure dataset.
    """

    def __call__(self, data: Dict, **kwargs: Any) -> Structure:
        ...

    def __repr__(self) -> str:
        ...


class ToStructure(Transformer):
    """Transform a dictionary of data into to a Structure."""

    def __init__(
        self,
        r_cutoff: Optional[float] = None,
        dtype: Optional[Dtype] = None,
    ) -> None:
        self.r_cutoff: Optional[float] = r_cutoff
        self.dtype: Dtype = _dtype.FLOATX if dtype is None else dtype

    def __call__(self, data: Dict[str, Any]) -> Structure:
        return Structure.create_from_dict(
            data,
            r_cutoff=self.r_cutoff,
            dtype=self.dtype,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
