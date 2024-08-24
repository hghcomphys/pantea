from functools import partial
from typing import Dict, Protocol

import jax.numpy as jnp
from frozendict import frozendict
from jax import jit

from pantea.atoms.structure import StructureInfo
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
    model_params: frozendict[str, Array],
    structure: StructureInfo,
) -> Array:
    """Compute model output per-atom energy."""
    x = _calculate_acsf_descriptor(
        atomic_potential.descriptor,
        positions,
        structure,
    )
    x = atomic_potential.scaler(x)
    x = atomic_potential.model.apply({"params": model_params}, x)  # type: ignore
    return x


# _jitted_compute_energy_per_atom = jit(_compute_energy_per_atom, static_argnums=(0,))


@partial(jit, static_argnums=(0,))
def _compute_energy(
    atomic_potential_dict: Dict[Element, AtomicPotentialInterface],
    positions_dict: Dict[Element, Array],
    params_dict: Dict[Element, frozendict[str, Array]],
    structure: StructureInfo,
) -> Array:
    """Calculate the total potential energy."""
    total_energy: Array = jnp.array(0.0)
    for element in atomic_potential_dict:
        per_atom_energy = _compute_energy_per_atom(
            atomic_potential_dict[element],
            positions_dict[element],
            params_dict[element],
            structure,
        )
        total_energy += jnp.sum(per_atom_energy)
    return total_energy
