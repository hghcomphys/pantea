from ..base import BaseTorchipClass
from torch import nn


class BaseModel(nn.Module, BaseTorchipClass):
    """
    A base class for all kinds of ML-based potential models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
