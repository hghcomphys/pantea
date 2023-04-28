import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class Logger:
    """
    An utility logger class which allows more tweaks than the standard python logging.
    """

    def __init__(self, level=logging.WARNING, filename=None) -> None:
        self.logger: Logger = logging.getLogger("Jaxip")  # type: ignore
        self.level: int = level
        self.handlers = list()

        self._add_console_handler()
        if filename:
            self._add_file_handler(filename)

        for handler in logging.root.handlers:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.DEBUG, handlers=self.handlers)

        # Avoiding matplotlib debug messages
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("matplotlib.font_manager").disabled = True
        logging.getLogger("jax").setLevel(logging.WARNING)

    def set_level(self, level) -> None:
        """
        Set the console logging level.

        :param level: INFO, WARNING, ERROR, and DEBUG
        """
        self.level = level
        logging.root.handlers[0].setLevel(self.level)

    def _add_console_handler(self) -> None:
        """
        Prepare console handler and adding it as a default handler.
        """
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(
            logging.Formatter("%(levelname)s %(message)s")
        )  # logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        console_handler.setLevel(self.level)
        self.handlers.append(console_handler)

    def _add_file_handler(self, filename: Path) -> None:
        """
        Add a file handler to the logging.

        :param filename: output log file name
        :type filename: Path
        """
        # File handler is also useful for multi-process logging.
        file_handler = RotatingFileHandler(str(Path(filename)), "a", 1_000_000, 3)
        file_handler.setFormatter(
            logging.Formatter(
                "%(levelname)-8s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
            )
        )
        file_handler.setLevel(logging.DEBUG)
        self.handlers.append(file_handler)

    def debug(self, msg, *args, **kwargs) -> None:
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs) -> None:
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs) -> None:
        self.logger.warning(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs) -> None:
        self.warning(msg, *args, **kwargs)

    def error(self, msg, exception: Any = None, *args, **kwargs) -> None:
        self.logger.error(msg, *args, **kwargs)
        if exception is not None:
            raise exception(msg)


# Create the global logger object
logger = Logger()  # filename="debug.log"


def set_logging_level(level: int) -> None:
    """
    Set the global logging level

    :param level: INFO, WARNING, ERROR, and DEBUG
    :type level: int (logging)
    """
    logger.set_level(level)


class LoggingContextManager:
    """
    This utility class allows to monitor logging messages with different level
    (e.g. INFO or DEBUG) for a particular section of the scripts.
    """

    def __init__(self, level) -> None:
        self.level = level
        self.current_level = None

    def __enter__(self) -> Logger:
        global logger
        self.current_level = logger.level
        logger.set_level(self.level)
        return logger

    def __exit__(self, type, value, traceback) -> None:
        global logger
        logger.set_level(self.current_level)
