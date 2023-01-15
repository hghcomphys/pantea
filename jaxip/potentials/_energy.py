from functools import partial
from typing import Dict

import jax.numpy as jnp
from frozendict import frozendict
from jax import grad, jit

from jaxip.descriptors.acsf._acsf import _calculate_descriptor
from jaxip.types import Array


@partial(jit, static_argnums=(0,))  # FIXME
def _energy_fn(
    atomic_potential: frozendict,
    positions: Dict[str, Array],
    params: Dict[str, frozendict],
    xbatch: Dict,
) -> Array:
    """
    A helper function that allows to calculate gradient of the NNP total energy
    respect to the atom positions (for each element).
    """
    # TODO: using jax.lax.scan?
    elements: list[str] = list(xbatch.keys())
    total_energy: Array = jnp.array(0.0)
    for element in elements:
        potential = atomic_potential[element]
        inputs = xbatch[element]
        x = _calculate_descriptor(
            potential.descriptor,
            positions[element],
            inputs.position,
            inputs.atype,
            inputs.lattice,
            inputs.emap,
        )
        x = potential.scaler(x)
        x = potential.model.apply({"params": params[element]}, x)
        total_energy += jnp.sum(x)

    return total_energy


_grad_energy_fn = jit(
    grad(_energy_fn, argnums=1),
    static_argnums=0,
)


# @partial(jit, static_argnums=(0,))  # FIXME
def _compute_force(
    atomic_potential: frozendict,
    positions: Dict[str, Array],
    params: Dict[str, frozendict],
    xbatch: Dict,
) -> Dict[str, Array]:
    """Compute force components using the gradient of the energy."""
    grad_energies: Dict[str, Array] = _grad_energy_fn(
        atomic_potential,
        positions,
        params,
        xbatch,
    )
    return {
        element: -1.0 * grad_energy for element, grad_energy in grad_energies.items()
    }
