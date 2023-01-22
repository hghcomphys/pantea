from typing import Any, List

from pydantic import BaseModel


class _CFG(BaseModel):
    """A base class for default values as global configuration."""

    # TODO: add methods to read configurations from dict and file inputs
    # Pydantic validators can be added to check attributes in subclasses

    def __getitem__(self, keyword: str) -> Any:
        """Get value for the input name."""
        return getattr(self, keyword)

    def __setitem__(self, name, value) -> None:
        """Set value for the input name"""
        setattr(self, name, value)

    def keywords(self) -> List[str]:
        """Return a list of existing keyword names."""
        return self.__dict__.keys()  # type: ignore
