from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Protocol

from pantea.atoms.structure import Structure
from pantea.datasets.runner import RunnerDataSource
from pantea.logger import logger
from pantea.types import Dtype


class DataSourceInterface(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> Structure:
        ...


@dataclass
class Dataset:
    """A container for Structure data with caching support."""

    data: DataSourceInterface
    persist: bool
    cache: Dict[int, Structure] = field(default_factory=dict, repr=False)

    @classmethod
    def from_runner(
        cls,
        filename: Path,
        persist: bool = False,
        dtype: Optional[Dtype] = None,
    ):
        dataset = RunnerDataSource(filename, dtype)
        return cls(dataset, persist)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Structure:
        """Read the desired structure, if possible from cache."""
        if self.persist and (index in self.cache):
            return self.cache[index]
        else:
            logger.debug(f"loading structure({index=})")
            structure = self.data[index]
            if self.persist:
                self.cache[index] = structure
            return structure

    def preload(self) -> None:
        """
        Preload (cache) all structures into memory.

        This ensures that any structure can be rapidly loaded from memory in subsequent operations.
        """
        logger.info("Preloading (caching) all structures")
        self.persist = True
        for index in range(len(self)):
            self.cache[index] = self.data[index]
