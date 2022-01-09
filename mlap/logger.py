import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger('MLAP') 
logger.setLevel(logging.DEBUG)

# File handler is also useful for multi-process logging.
file_handler = RotatingFileHandler('mlap.log', 'a', 1e6, 3)
file_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(processName)-10s] %(name)s %(levelname)-8s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p'))
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p'))
console_handler.setLevel(logging.INFO)

# logger.addHandler(file_handler)
logger.addHandler(console_handler)
#logging.getLogger('matplotlib').setLevel(logging.ERROR)  # you don't want the matplotlib debug logs