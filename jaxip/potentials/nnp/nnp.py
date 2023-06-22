from typing import Dict, Protocol, Tuple

from frozendict import frozendict

from jaxip.potentials.atomic_potential import AtomicPotential
from jaxip.potentials.nnp.settings import \
    NeuralNetworkPotentialSettings as PotentialSettings
from jaxip.types import Element


class NeuralNetworkPotentialInterface(Protocol):
    """Interface for neural network potential (NNP)."""

    settings: PotentialSettings
    elements: Tuple[Element, ...]
    atomic_potential: Dict[Element, AtomicPotential]
    model_params: Dict[Element, frozendict]
