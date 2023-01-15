from functools import partial
from typing import Protocol

import jax
from frozendict import frozendict

from jaxip.descriptors import DescriptorScaler
from jaxip.descriptors.acsf._acsf import _calculate_descriptor
from jaxip.descriptors.base import Descriptor
from jaxip.models import NeuralNetworkModel
from jaxip.structure.structure import Inputs
from jaxip.types import Array


class AtomicPotentialInterface(Protocol):
    """An interface for atomic potential."""

    descriptor: Descriptor
    scaler: DescriptorScaler
    model: NeuralNetworkModel


@partial(jax.jit, static_argnums=(0,))  # FIXME
def _compute_atomic_energy(
    atomic_potential: AtomicPotentialInterface,
    positions: Array,
    params: frozendict,
    inputs: Inputs,
) -> Array:
    """Compute model output energy."""
    x = _calculate_descriptor(
        atomic_potential.descriptor,
        positions,
        inputs.position,
        inputs.atype,
        inputs.lattice,
        inputs.emap,
    )
    x = atomic_potential.scaler(x)
    atomic_energy = atomic_potential.model.apply({"params": params}, x)
    return atomic_energy  # type: ignore
