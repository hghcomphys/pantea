from .logger import logger


class BaseTorchip:
    """
    A base class which all other classes have to be derived from this one.
    It's basically intended to be used for setting the global properties or methods
    which expected to exist in all child classes.
    """

    def __init__(self):
        logger.debug(f"Initializing {self}")

    def __repr__(self) -> str:
        return "{C}({attrs})".format(  # @{id:x}
            C=self.__class__.__name__,
            # id=id(self) & 0xFFFFFF,
            attrs=", ".join(
                "{}={!r}".format(k, v)
                for k, v in self.__dict__.items()
                if not k.startswith("_")
            ),
        )
