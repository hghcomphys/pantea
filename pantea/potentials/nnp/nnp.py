from typing import Dict, Protocol, Tuple

from frozendict import frozendict
from pantea.potentials.atomic_potential import AtomicPotential
from pantea.potentials.nnp.settings import (
    NeuralNetworkPotentialSettings as PotentialSettings,
)
from pantea.types import Element


class NeuralNetworkPotentialInterface(Protocol):
    """Interface for neural network potential (NNP)."""

    settings: PotentialSettings
    elements: Tuple[Element, ...]
    atomic_potential: Dict[Element, AtomicPotential]
    model_params: Dict[Element, frozendict]
