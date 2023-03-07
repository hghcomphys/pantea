from typing import Dict, Protocol

from jaxip.datasets.base import StructureDataset


class Updater(Protocol):
    """An interface for potential updaters."""

    def fit(self, dataset: StructureDataset) -> Dict:
        ...

