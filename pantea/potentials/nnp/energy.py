from functools import partial
from typing import Dict, Protocol

from jax import jit

from pantea.atoms.structure import StructureAsKernelArgs
from pantea.descriptors import DescriptorScaler
from pantea.descriptors.acsf.acsf import ACSF, _calculate_acsf_descriptor
from pantea.descriptors.scaler import ScalerParams
from pantea.models import NeuralNetworkModel
from pantea.models.nn.model import ModelParams
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
    model_params: ModelParams,
    scaler_params: ScalerParams,
    structure: StructureAsKernelArgs,
) -> Array:
    """Compute model output per-atom energy."""
    x = _calculate_acsf_descriptor(
        atomic_potential.descriptor,
        positions,
        structure,
    )
    x = atomic_potential.scaler(scaler_params, x)
    x = atomic_potential.model.apply({"params": model_params}, x)  # type: ignore
    return x


_jitted_compute_energy_per_atom = jit(_compute_energy_per_atom, static_argnums=(0,))


def _compute_energy(
    atomic_potentials: Dict[Element, AtomicPotentialInterface],
    positions: Dict[Element, Array],
    models_params: Dict[Element, ModelParams],
    scalers_params: Dict[Element, ScalerParams],
    structure: StructureAsKernelArgs,
) -> Array:
    """Calculate the total potential energy."""
    total_energy = 0.0
    for element in atomic_potentials:
        energies: Array = _compute_energy_per_atom(
            atomic_potentials[element],
            positions[element],
            models_params[element],
            scalers_params[element],
            structure,
        )
        total_energy += energies.sum()
    return total_energy


_jitted_compute_energy = jit(_compute_energy, static_argnums=(0,))
