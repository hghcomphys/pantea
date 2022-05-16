"""
Setup logger including the console and the rotating file handlers.
"""

# from .config import CFG
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger('TorchIP') 
logger.setLevel(logging.DEBUG)

# File handler is also useful for multi-process logging.
file_handler = RotatingFileHandler("torchip.log", 'a', 1e6, 3)
file_handler.setFormatter(logging.Formatter('%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')) 
# %(levelname)-8s [%(asctime)s]  [%(processName)-10s] %(name)s 
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(message)s'))
# logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
console_handler.setLevel(logging.WARNING)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
#logging.getLogger('matplotlib').setLevel(logging.ERROR)  # we don't want matplotlib logs
