from ..base import _Base
from torch.utils.data import Dataset


class StructureDataset(Dataset, _Base):  # FIXME: multiple inheritance
    """
    A base class for atomic data structure in a lazy mode.
    All future atomic datasets must be derived from this class.
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError
