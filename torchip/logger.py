# from .config import CFG
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger('MLP-Framework') 
logger.setLevel(logging.DEBUG)

# File handler is also useful for multi-process logging.
file_handler = RotatingFileHandler("log.out", 'a', 1e6, 3)
file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)-8s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')) # [%(processName)-10s] %(name)s 
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
# console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p'))
console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
console_handler.setLevel(logging.WARNING)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
#logging.getLogger('matplotlib').setLevel(logging.ERROR)  # we don't want matplotlib logs
