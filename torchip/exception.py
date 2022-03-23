from .logger import logger


class CustomErrorException(Exception):
  """
  A customized exception that logs the error message prior to throwing the exception.
  """
  def __init__(self, message="Somthing wrong"):
    logger.error(message)
    super().__init__(message)