from ..base import _Base


class BaseModel(_Base):
    """
    A base class for all kinds of ML-based potential models.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
