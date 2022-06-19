"""
Setup logger including the console and the rotating file handlers.
"""

# from .config import CFG
import sys
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler


class Logger:
  """
  An utility logger class which allows more tweaks than the standard python logging.  
  """
  def __init__(self, level=logging.WARNING, filename=None):
    self.logger = logging.getLogger("TorchIP") 
    self.level = level
    self.handlers = []

    self._add_console_handler()
    if filename:
      self._add_file_handler(filename)
   
    for handler in logging.root.handlers:
      logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG, handlers=self.handlers)

    # Avoiding matplotlib debug messages
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').disabled = True

  def set_level(self, level) -> None:
    """
    Set the console logging level.  

    Args:
        level (logging): INFO, WARNING, ERROR, and DEBUG
    """
    self.level = level
    logging.root.handlers[0].setLevel(self.level)

  def _add_console_handler(self) -> None:
    """
    Prepare console handler and adding it as a default handler. 
    Further fine-tune adjustments (e.g. formatting) on console handler can be applied here. 
    """
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter('%(message)s'))  # logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    console_handler.setLevel(self.level)
    self.handlers.append(console_handler)

  def _add_file_handler(self, filename) -> None:
    """
    Add a file handler to the logging.
    Further fine-tune adjustments (e.g. formatting) on file handler can be applied here. 

    Args:
        logfile (str, optional): Path to log file name. Defaults to "debug.log".
        level (logging, optional): Logging level for the log file. Defaults to logging.DEBUG.
    """
    # File handler is also useful for multi-process logging.
    file_handler = RotatingFileHandler(str(Path(filename)), 'a', 1e6, 3)
    file_handler.setFormatter(logging.Formatter('%(levelname)-8s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p'))  #  [%(asctime)s] [%(threadName)-10s]  [%(processName)-10s] %(name)s 
    file_handler.setLevel(logging.DEBUG)
    self.handlers.append(file_handler)

  def debug(self, msg, *args, **kwargs):
    self.logger.debug(msg, *args, **kwargs)
  
  def print(self, msg="", **kwargs):
    # TODO: ignore printing if current level is >= INFO
    print(msg, **kwargs)
    self.info(msg)

  def info(self, msg, *args, **kwargs):
    self.logger.info(msg, *args, **kwargs)
  
  def warning(self, msg, *args, **kwargs):
    self.logger.warning(msg, *args, **kwargs)

  def warn(self, msg, *args, **kwargs):
    self.warning(msg, *args, **kwargs)

  def error(self, msg, *args, **kwargs):
    # TODO: get exception type as input argument
    self.logger.error(msg, *args, **kwargs)
    raise Exception(msg)


# Create a global logger object
logger = Logger() # filename="debug.log"
