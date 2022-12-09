from typing import Any, Dict
from mlpot.logger import logger


class _CFG:
    """
    A configuration class of default values for global variables.
    """

    # TODO: circular import error between CFG & logger
    _conf: Dict[str, Any] = {
        # Global variables!
    }

    def __init__(self) -> None:
        self._set_attributes()

    def _set_attributes(self) -> None:
        for atr in self._conf:
            setattr(self, atr, self._conf[atr])

    def get(self, name) -> Any:
        return self._conf[name]

    def set(self, name, value) -> None:
        if name in self._conf.keys():
            logger.debug(f"Setting {name}: '{value}'")
            self._conf[name] = value
        else:
            logger.error(
                f"Name '{name}' is not allowed!",
                exception=NameError,
            )

    def __getitem__(self, name) -> Any:
        return self.get(name)
