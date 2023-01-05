from dataclasses import dataclass
from typing import Any

from jaxip.logger import logger


@dataclass
class _CFG:
    """A base configuration class of default values for global variables."""

    # TODO: add methods to read configurations from dict and file inputs

    def get(self, name: str) -> Any:
        return getattr(self, name)

    def set(self, name: str, value: Any) -> None:
        if name in self.__dict__:
            logger.debug(f"Setting {name}: '{value}'")
            setattr(self, name, value)
        else:
            logger.error(
                f"Name '{name}' is not allowed!",
                exception=NameError,
            )

    def __getitem__(self, name) -> Any:
        return self.get(name)
