from functools import partial
from typing import Callable, Dict, Protocol, Tuple

import jax.numpy as jnp
from jax import jit, lax, vmap

from mlpot.descriptors.acsf.angular import AngularSymmetryFunction
from mlpot.descriptors.acsf.radial import RadialSymmetryFunction
from mlpot.descriptors.acsf.symmetry import EnvironmentElements
from mlpot.structure._neighbor import _calculate_cutoff_mask_per_atom
from mlpot.structure._structure import _calculate_distance_per_atom
from mlpot.types import Array


class AcsfInterface(Protocol):
    num_symmetry_functions: int
    num_radial_symmetry_functions: int
    num_angular_symmetry_functions: int
    radial_symmetry_functions: Tuple
    angular_symmetry_functions: Tuple


@jit
def _calculate_radial_acsf_per_atom(
    radial: Dict[EnvironmentElements, RadialSymmetryFunction],
    atype: Array,
    dist_i: Array,
    emap: Dict[str, Array],
) -> Array:

    elements: EnvironmentElements = [k for k in radial.keys()][0]

    # cutoff-radius mask
    mask_cutoff_i = _calculate_cutoff_mask_per_atom(
        dist_i, jnp.asarray(radial[elements].r_cutoff)
    )
    # get the corresponding neighboring atom types
    mask_cutoff_and_atype_ij = mask_cutoff_i & (atype == emap[elements.neighbor_j])

    return jnp.sum(
        radial[elements](dist_i),
        where=mask_cutoff_and_atype_ij,
        axis=0,
    )


@jit
def _calculate_angular_acsf_per_atom(
    angular: Dict[EnvironmentElements, AngularSymmetryFunction],
    atype: Array,
    diff_i: Array,
    dist_i: Array,
    lattice: Array,
    emap: Dict[str, Array],
) -> Array:

    elements: EnvironmentElements = [k for k in angular.keys()][0]

    # cutoff-radius mask
    mask_cutoff_i = _calculate_cutoff_mask_per_atom(
        dist_i, jnp.asarray(angular[elements].r_cutoff)
    )
    # mask for neighboring element j
    at_j = emap[elements.neighbor_j]
    mask_cutoff_and_atype_ij = mask_cutoff_i & (atype == at_j)
    # mask for neighboring element k
    at_k = emap[elements.neighbor_k]  # type: ignore
    mask_cutoff_and_atype_ik = mask_cutoff_i & (atype == at_k)

    total, _ = lax.scan(
        partial(
            _inner_loop_over_angular_acsf_terms,
            mask_ik=mask_cutoff_and_atype_ik,
            diff_i=diff_i,
            dist_i=dist_i,
            lattice=lattice,
            kernel=angular[elements],
        ),
        jnp.asarray(0.0),
        (diff_i, dist_i, mask_cutoff_and_atype_ij),
    )
    # correct double-counting
    total = lax.cond(
        at_j == at_k,
        lambda x: x * 0.5,
        lambda x: x,
        total,
    )
    return total


# Called by jax.lax.scan (no need for @jax.jit)
def _inner_loop_over_angular_acsf_terms(
    total: Array,
    inputs: Tuple[Array, ...],
    diff_i: Array,
    dist_i: Array,
    mask_ik: Array,
    lattice: Array,
    kernel: Callable,
) -> Tuple[Array, Array]:

    # Scan occurs along the leading axis
    Rij, rij, mask_ij = inputs

    # rik = jnp.where(
    #     mask_cutoff_and_atype_ik,
    #     dist_i,
    #     0.0,
    # ) # same for Rjk if using diff_i

    # fix nan issue in gradient
    # see https://github.com/google/jax/issues/1052#issuecomment-514083352
    operand = rij * dist_i
    is_zero = operand == 0.0
    true_op = jnp.where(is_zero, 1.0, operand)
    cost = jnp.where(
        mask_ik,
        jnp.inner(Rij, diff_i) / true_op,
        1.0,
    )
    cost = jnp.where(is_zero, 0.0, cost)  # FIXME: gradient value!

    rjk = jnp.where(  # diff_jk = diff_ji - diff_ik
        mask_ik,
        _calculate_distance_per_atom(Rij, diff_i, lattice)[0],
        0.0,
    )  # second tuple output for Rjk

    value = jnp.where(
        mask_ij,
        jnp.sum(
            kernel(rij, dist_i, rjk, cost),
            where=mask_ik & (rjk > 0.0),  # exclude k=j
            axis=0,
        ),
        0.0,
    )
    return total + value, value


@jit
def _calculate_descriptor_per_atom(
    acsf: AcsfInterface,
    atom_position: Array,  # must be a single atom position shape=(1, 3)
    neighbor_positions: Array,
    atype: Array,
    lattice: Array,
    emap: Dict,
) -> Array:
    """
    Compute descriptor values per atom in the structure (via atom id).
    """
    dtype = atom_position.dtype
    result = jnp.empty(acsf.num_symmetry_functions, dtype=dtype)

    dist_i, diff_i = _calculate_distance_per_atom(
        atom_position, neighbor_positions, lattice
    )

    # Loop over the radial terms
    for index, (elements, radial) in enumerate(acsf.radial_symmetry_functions):
        result = result.at[index].set(
            _calculate_radial_acsf_per_atom({elements: radial}, atype, dist_i, emap)
        )

    # Loop over the angular terms
    for index, (elements, angular) in enumerate(
        acsf.angular_symmetry_functions, start=acsf.num_radial_symmetry_functions
    ):
        result = result.at[index].set(
            _calculate_angular_acsf_per_atom(
                {elements: angular}, atype, diff_i, dist_i, lattice, emap
            )
        )

    return result


_calculate_descriptor = jit(
    vmap(
        _calculate_descriptor_per_atom,
        in_axes=(None, 0, None, None, None, None),
    ),
    static_argnums=(0,),
)
