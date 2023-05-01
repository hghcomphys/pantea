from typing import Dict, Protocol, Tuple

from frozendict import frozendict

from jaxip.config import _CFG as PotentialSettings
from jaxip.datasets.base import StructureDataset
from jaxip.potentials.atomic_potential import AtomicPotential
from jaxip.types import Element


class Updater(Protocol):
    """Interface for potential weight updaters."""

    def fit(self, dataset: StructureDataset) -> Dict:
        ...


class Potential(Protocol):
    """Interface for Potential."""

    settings: PotentialSettings
    elements: Tuple[Element, ...]
    atomic_potential: Dict[Element, AtomicPotential]
    model_params: Dict[Element, frozendict]
