import jax.numpy as jnp
from frozendict import frozendict
from typing import Dict
from functools import partial
from jax import jit, grad
from mlpot.descriptors.asf._asf import _calculate_descriptor
from mlpot.types import Array


@partial(jit, static_argnums=(0,))  # FIXME
def _energy_fn(
    static_args: Dict,
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
    energy: Array = jnp.array(0.0)

    for element in elements:

        static_arg = static_args[element]
        input = xbatch[element]

        dsc = _calculate_descriptor(
            static_arg.descriptor,
            positions[element],
            input.position,
            input.atype,
            input.lattice,
            input.emap,
        )
        dsc = static_arg.scaler(dsc)
        energy += jnp.sum(static_arg.model.apply({"params": params[element]}, dsc))

    return energy


_grad_energy_fn = jit(
    grad(_energy_fn, argnums=1),
    static_argnums=0,
)


# @partial(jit, static_argnums=(0,))  # FIXME
def _compute_forces(
    static_input: Dict,
    positions: Dict[str, Array],
    params: Dict[str, frozendict],
    xbatch: Dict,
) -> Dict[str, Array]:
    grad_energies: Dict[str, Array] = _grad_energy_fn(
        static_input,
        positions,
        params,
        xbatch,
    )
    return {
        element: -1.0 * grad_energy for element, grad_energy in grad_energies.items()
    }
