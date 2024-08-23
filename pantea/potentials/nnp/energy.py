from functools import partial
from typing import Dict, Protocol

import jax.numpy as jnp
from frozendict import frozendict
from jax import jit

from pantea.atoms.structure import Inputs
from pantea.descriptors import DescriptorScaler
from pantea.descriptors.acsf.acsf import ACSF, _calculate_acsf_descriptor
from pantea.models import NeuralNetworkModel
from pantea.types import Array, Element


class AtomicPotentialInterface(Protocol):
    """An interface for AtomicPotential."""

    descriptor: ACSF
    scaler: DescriptorScaler
    model: NeuralNetworkModel


@partial(jit, static_argnums=(0,))
def _compute_energy_per_atom(
    atomic_potential: AtomicPotentialInterface,
    positions: Array,
    params: frozendict[str, Array],
    inputs: Inputs,
) -> Array:
    """Compute model output per-atom energy."""
    x = _calculate_acsf_descriptor(
        atomic_potential.descriptor,
        positions,
        inputs.positions,
        inputs.atom_types,
        inputs.lattice,
        inputs.element_map,
    )
    x = atomic_potential.scaler(x)
    atomic_energy = atomic_potential.model.apply({"params": params}, x)  # type: ignore
    return atomic_energy


# _jitted_compute_energy_per_atom = jit(_compute_energy_per_atom, static_argnums=(0,))


@partial(jit, static_argnums=(0,))
def _compute_energy(
    atomic_potential: Dict[Element, AtomicPotentialInterface],
    positions: Dict[Element, Array],
    params: Dict[Element, frozendict[str, Array]],
    inputs: Dict[Element, Inputs],
) -> Array:
    """
    A helper function that allows to calculate gradient of the NNP total energy
    respect to the atom positions (for each element).
    """
    elements: list[Element] = list(inputs.keys())
    total_energy: Array = jnp.array(0.0)
    for element in elements:
        per_atom_energy = _compute_energy_per_atom(
            atomic_potential[element],
            positions[element],
            params[element],
            inputs[element],
        )
        total_energy += jnp.sum(per_atom_energy)
    return total_energy
