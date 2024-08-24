from functools import partial
from typing import Dict

import jax
from frozendict import frozendict
from jax import grad, jit

from pantea.atoms.structure import StructureInfo
from pantea.potentials.nnp.energy import AtomicPotentialInterface, _compute_energy
from pantea.types import Array, Element

_grad_compute_energy = grad(_compute_energy, argnums=1)

_jitted_grad_compute_energy = jit(
    _grad_compute_energy,
    static_argnums=0,
)


def negative(array: Array) -> Array:
    return -1.0 * array


# @partial(jit, static_argnums=(0,))
def _compute_forces(
    atomic_potential_dict: Dict[Element, AtomicPotentialInterface],
    positions_dict: Dict[Element, Array],
    params_dict: Dict[Element, frozendict],
    structure: StructureInfo,
) -> Dict[Element, Array]:
    """Compute force components using the gradient of the total energy."""
    energy_gradients = _jitted_grad_compute_energy(
        atomic_potential_dict,
        positions_dict,
        params_dict,
        structure,
    )
    return jax.tree.map(negative, energy_gradients)
