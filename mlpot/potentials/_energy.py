import jax.numpy as jnp
from frozendict import frozendict
from typing import Tuple
from functools import partial
from jax import jit, grad
from mlpot.descriptors.asf._asf import _calculate_descriptor
from mlpot.types import Array


@partial(jit, static_argnums=(0,))  # FIXME
def _energy_fn(
    sargs: Tuple,
    xs: Tuple[jnp.ndarray],
    params: Tuple[frozendict],
    xbatch: Tuple,
) -> Array:
    """
    A helper function that allows to calculate gradient of the NNP total energy
    respect to the atom positions (for each element).
    """
    # TODO: using jax.lax.scan?
    energy = jnp.array(0.0)
    for p, inputs, static_inputs, x in zip(params, xbatch, sargs, xs):
        _, position, atype, lattice, emap = inputs
        asf, scaler, model = static_inputs

        dsc = _calculate_descriptor(asf, x, position, atype, lattice, emap)
        scaled_dsc = scaler(dsc)
        energy += jnp.sum(model.apply({"params": p}, scaled_dsc))

    return energy


_grad_energy_fn = jit(
    grad(_energy_fn, argnums=1),
    static_argnums=0,
)


@partial(jit, static_argnums=(0,))  # FIXME
def _compute_forces(
    sargs: Tuple,
    xs: Tuple[jnp.ndarray],
    params: Tuple[frozendict],
    xbatch: Tuple,
) -> Tuple[jnp.ndarray]:
    grad_energies = _grad_energy_fn(sargs, xs, params, xbatch)
    return tuple(-1 * grad_energy for grad_energy in grad_energies)
