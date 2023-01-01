# from ...structure._structure import _calculate_distance_per_atom
# from .angular import AngularSymmetryFunction
# from .radial import RadialSymmetryFunction
# from ._asf import _calculate_radial_asf_per_atom
# from ._asf import _calculate_angular_asf_per_atom
# from typing import Tuple, Dict
# from functools import partial
# from jax import jit, grad, vmap
# import jax.numpy as jnp


# A helper function to calculate gradient of radial ASF
# @partial(jit, static_argnums=(0,))  # FIXME
# def _func_asf_radial(
#     radial: Tuple[RadialSymmetryFunction, str, str],
#     x: jnp.ndarray,
#     aid: jnp.ndarray,
#     position: jnp.ndarray,
#     lattice: jnp.ndarray,
#     atype: jnp.ndarray,
#     emap: Dict,
# ) -> jnp.ndarray:

#     dist_i, _ = _calculate_distance_per_atom(x, position, lattice)

#     return _calculate_radial_asf_per_atom(radial, aid, atype, dist_i, emap)


# # A help function to calculate gradient of angular ASF respect to x
# @partial(jit, static_argnums=(0,))  # FIXME
# def _func_asf_angular(
#     angular: Tuple[AngularSymmetryFunction, str, str, str],
#     x: jnp.ndarray,
#     aid: jnp.ndarray,
#     position: jnp.ndarray,
#     lattice: jnp.ndarray,
#     atype: jnp.ndarray,
#     emap: Dict,
# ) -> jnp.ndarray:

#     dist_i, diff_i = _calculate_distance_per_atom(x, position, lattice)

#     return _calculate_angular_asf_per_atom(
#         angular, aid, atype, diff_i, dist_i, lattice, emap
#     )


# _grad_func_asf_radial = jit(
#     grad(_func_asf_radial, argnums=1),
#     static_argnums=(0,),
# )

# _grad_func_asf_angular = jit(
#     grad(_func_asf_angular, argnums=1),
#     static_argnums=(0,),
# )


# # TODO: jax.jit
# def _grad_func_asf(
#     acsf,
#     asf_index: int,
#     aid: jnp.ndarray,
#     position: jnp.ndarray,
#     lattice: jnp.ndarray,
#     atype: jnp.ndarray,
#     emap: Dict[str, int],
# ) -> jnp.ndarray:

#     if asf_index < acsf.n_radial_symmetry_functions:
#         grad_value = _grad_func_asf_radial(
#             acsf._radial[asf_index],
#             position[aid],
#             aid,
#             position,
#             lattice,
#             atype,
#             emap,
#         )
#     else:
#         grad_value = _grad_func_asf_angular(
#             acsf._angular[asf_index - acsf.n_radial_symmetry_functions],
#             position[aid],
#             aid,
#             position,
#             lattice,
#             atype,
#             emap,
#         )

#     return jnp.squeeze(grad_value)


# _vmap_grad_func_asf = vmap(
#     _grad_func_asf,
#     in_axes=(None, None, 0, None, None, None, None),
# )

# # FIXME: nested vmap over asf_index
# # _vmap2_grad_func_asf = vmap(
# #     _vmap_grad_func_asf,
# #     in_axes=(None, None, 0, None, None, None, None),
# # )
