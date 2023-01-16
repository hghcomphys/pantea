from typing import Dict

from frozendict import frozendict
from jax import grad, jit

from jaxip.potentials._energy import AtomicPotentialInterface, _energy_fn
from jaxip.structure.structure import Inputs
from jaxip.types import Array, Element

_grad_energy_fn = jit(
    grad(_energy_fn, argnums=1),
    static_argnums=0,
)


# @partial(jit, static_argnums=(0,))  # FIXME
def _compute_force(
    atomic_potential: Dict[Element, AtomicPotentialInterface],
    positions: Dict[Element, Array],
    params: Dict[Element, frozendict],
    xbatch: Dict[Element, Inputs],
) -> Dict[str, Array]:
    """Compute force components using the gradient of the total energy."""
    grad_energies: Dict[Element, Array] = _grad_energy_fn(
        atomic_potential,
        positions,
        params,
        xbatch,
    )
    return {
        element: -1.0 * grad_energy for element, grad_energy in grad_energies.items()
    }