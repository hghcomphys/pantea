from ...structure._structure import _calculate_distance_per_atom
from ._asf import _calculate_radial_asf_per_atom
from ._asf import _calculate_angular_asf_per_atom
from .angular import AngularSymmetryFunction
from .radial import RadialSymmetryFunction
from typing import Tuple, Dict
import jax
import jax.numpy as jnp


Tensor = jnp.ndarray


def _func_asf_radial(
    radial: Tuple[RadialSymmetryFunction, str, str],
    x: Tensor,
    aid: Tensor,
    position: Tensor,
    lattice: Tensor,
    atype: Tensor,
    emap: Dict,
) -> Tensor:

    dist_i, _ = _calculate_distance_per_atom(x, position, lattice)

    return _calculate_radial_asf_per_atom(radial, aid, atype, dist_i, emap)


def _func_asf_angular(
    angular: Tuple[AngularSymmetryFunction, str, str, str],
    x: Tensor,
    aid: Tensor,
    position: Tensor,
    lattice: Tensor,
    atype: Tensor,
    emap: Dict,
) -> Tensor:

    dist_i, diff_i = _calculate_distance_per_atom(x, position, lattice)

    return _calculate_angular_asf_per_atom(
        angular, aid, atype, diff_i, dist_i, lattice, emap
    )


_grad_asf_func_radial = jax.jit(
    jax.grad(_func_asf_radial, argnums=1),
    static_argnums=(0,),
)

_grad_asf_func_angular = jax.jit(
    jax.grad(_func_asf_angular, argnums=1),
    static_argnums=(0,),
)
