from typing import Dict, Protocol, Tuple

from frozendict import frozendict

from jaxip.potentials.atomic_potential import AtomicPotential
from jaxip.potentials.settings import PotentialSettings
from jaxip.types import Element


class Potential(Protocol):
    """Potential interface"""
    settings: PotentialSettings
    elements: Tuple[Element, ...]
    atomic_potential: Dict[Element, AtomicPotential]
    model_params: Dict[Element, frozendict]
