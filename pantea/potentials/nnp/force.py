from typing import Dict

from frozendict import frozendict
from jax import grad, jit

from pantea.atoms.structure import Inputs
from pantea.potentials.nnp.energy import AtomicPotentialInterface, _compute_energy
from pantea.types import Array, Element

_grad_compute_energy = jit(
    grad(_compute_energy, argnums=1),
    static_argnums=0,
)


# @partial(jit, static_argnums=(0,))
def _compute_forces(
    atomic_potential: Dict[Element, AtomicPotentialInterface],
    positions: Dict[Element, Array],
    params: Dict[Element, frozendict],
    inputs: Dict[Element, Inputs],
) -> Dict[Element, Array]:
    """Compute force components using the gradient of the total energy."""
    energy_gradients: Dict[Element, Array] = _grad_compute_energy(
        atomic_potential,
        positions,
        params,
        inputs,
    )
    return {
        element: -1.0 * energy_gradient
        for element, energy_gradient in energy_gradients.items()
    }
