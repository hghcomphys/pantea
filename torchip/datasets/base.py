from ..base import BaseTorchip
from torch.utils.data import Dataset


class StructureDataset(BaseTorchip, Dataset):
    """
    A base class for atomic data structure in a lazy mode.
    All future atomic datasets must be derived from this class.
    """

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError
