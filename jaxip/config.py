from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from pydantic import BaseModel


class _CFG(BaseModel):
    """A base class for default values as global configuration."""

    # Pydantic validators can be added to check attributes
    def __getitem__(self, keyword: str) -> Any:
        """Get value for the input name."""
        return getattr(self, keyword)

    def __setitem__(self, name, value) -> None:
        """Set value for the input name"""
        setattr(self, name, value)

    def keywords(self) -> List[str]:
        """Return a list of existing keyword names."""
        return self.__dict__.keys()  # type: ignore

    def to_json(self, file: Path) -> None:
        """Dump configuration into a json file."""
        with open(str(Path(file)), "w") as fp:
            json.dump(self.dict(), fp, indent=4)

    @classmethod
    def from_json(cls, file: Path) -> _CFG:
        """Create a configuration instance from the input json file."""
        with open(str(Path(file)), "r") as fp:
            kwargs = json.load(fp)
        return cls(**kwargs)
