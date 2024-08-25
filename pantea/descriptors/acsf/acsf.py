import itertools
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple

import jax.numpy as jnp
from jax import jacfwd, jit, lax, vmap

from pantea.atoms.distance import (
    _calculate_distances_per_atom,
    _calculate_distances_with_aux_per_atom,
)
from pantea.atoms.neighbor import _calculate_cutoff_masks_per_atom
from pantea.atoms.structure import Structure, StructureAsKernelArgs
from pantea.descriptors.acsf.angular import AngularSymmetryFunction
from pantea.descriptors.acsf.radial import RadialSymmetryFunction
from pantea.descriptors.acsf.symmetry import NeighborElements
from pantea.logger import logger
from pantea.pytree import BaseJaxPytreeDataClass, register_jax_pytree_node
from pantea.types import Array

AssignedRadialSymmetryFunction = Tuple[RadialSymmetryFunction, NeighborElements]
AssignedAngularSymmetryFunction = Tuple[AngularSymmetryFunction, NeighborElements]


@dataclass
class AtomCenteredSymmetryFunction(BaseJaxPytreeDataClass):
    """
    Atom-centered Symmetry Function (`ACSF`_) descriptor captures
    information about the distribution of neighboring
    atoms around a central atom by considering both radial (two-body) and angular
    (three-body) symmetry functions within a cutoff distance.
    Radial symmetry functions describe the distances,
    while angular symmetry functions hold information about the angles formed
    by the central atom with pairs of neighboring atoms.

    The ACSF represents a fingerprint of the local atomic environment and can
    be used in various machine learning potentials.

    .. _ACSF: https://compphysvienna.github.io/n2p2/topics/descriptors.html?highlight=symmetry%20function#
    """

    central_element: str
    radial_symmetry_functions: Tuple[AssignedRadialSymmetryFunction, ...]
    angular_symmetry_functions: Tuple[AssignedAngularSymmetryFunction, ...]

    def __call__(
        self,
        structure: Structure,
        atom_index: Optional[Array] = None,
    ) -> Array:
        """
        Calculate descriptor values for the input given structure and atom index.

        :param structure: input structure instance
        :type structure: Structure
        :param atom_index: atom indices, defaults select all atom indices
        of type the central element of the descriptor.
        :type atom_index: Optional[Array], optional
        :return: descriptor values
        :rtype: Array
        """
        if self.num_symmetry_functions == 0:
            logger.warning("No symmetry function was found")

        if atom_index is None:
            index = structure.select(self.central_element)
        else:
            index = jnp.atleast_1d(atom_index)
            # Check all atom types match the central element
            if not jnp.all(
                structure.element_map.element_to_atom_type[self.central_element]
                == structure.atom_types[index]
            ):
                logger.error(
                    f"Inconsistent central element '{self.central_element}': "
                    f" input atom index={atom_index}"
                    f" (atom_types='{int(structure.atom_types[index])}')",
                    exception=ValueError,
                )

        return _jitted_calculate_acsf_descriptor(
            self,
            structure.positions[index],
            structure.as_kernel_args(),
        )  # type: ignore

    def grad(
        self,
        structure: Structure,
        atom_index: Optional[Array] = None,
    ) -> Array:
        """
        Compute gradient of the ACSF descriptor respect to the atom position.

        :param structure: input Structure instance
        :param atom_index: atom index in Structure [0, natoms)
        :return: gradient of the descriptor value respect to the atom position

        Please note that `grad_per_element` method is way much faster than
        the current implementation of this method method.
        """
        if atom_index is None:
            positions = structure.positions
        else:
            index = jnp.atleast_1d(atom_index)
            # check atom index
            if not jnp.all((0 <= index) & (index < structure.natoms)):
                logger.error(
                    f"unexpected {atom_index=}."
                    f"Input index must be between [0, {structure.natoms})",
                    ValueError,
                )
            positions = structure.positions[index]

        return _jitted_calculate_grad_acsf_descriptor(
            self,
            positions,
            structure.as_args(),
        )  # type: ignore

    @property
    def num_radial_symmetry_functions(self) -> int:
        """Return number of `two-body` (radial) symmetry functions."""
        return len(self.radial_symmetry_functions)

    @property
    def num_angular_symmetry_functions(self) -> int:
        """Return number of `three-body` (angular) symmetry functions."""
        return len(self.angular_symmetry_functions)

    @property
    def num_symmetry_functions(self) -> int:
        """Return the total (`two-body` and `tree-body`) number of symmetry functions."""
        return self.num_radial_symmetry_functions + self.num_angular_symmetry_functions

    @property
    def r_cutoff(self) -> float:  # type: ignore
        """Return the maximum cutoff radius for list of the symmetry functions."""
        return max(
            [
                symmetry_function.r_cutoff
                for (symmetry_function, _) in itertools.chain(
                    *[
                        self.radial_symmetry_functions,
                        self.angular_symmetry_functions,
                    ]
                )
            ]
        )

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(central_element='{self.central_element}'"
            f", num_symmetry_functions={self.num_symmetry_functions})"
        )


def _calculate_acsf_descriptor_per_atom(
    acsf: AtomCenteredSymmetryFunction,
    position: Array,
    structure: StructureAsKernelArgs,
) -> Array:
    """Compute the ACSF descriptor values per atom."""
    result: Array = jnp.empty(acsf.num_symmetry_functions, dtype=position.dtype)
    # calculate distances respect to the reference atom (_i)
    distances_i, position_differences_i = _calculate_distances_with_aux_per_atom(
        position, structure.positions, structure.lattice
    )
    # Loop over the radial terms
    for index, (symmetry_function, assigned_elements) in enumerate(
        acsf.radial_symmetry_functions
    ):
        result = result.at[index].set(
            _calculate_radial_acsf_per_atom(
                symmetry_function,
                distances_i,
                structure.atom_types,
                structure.element_map[assigned_elements.neighbor_j],
            )
        )
    # Loop over the angular terms
    for index, (symmetry_function, assigned_elements) in enumerate(
        acsf.angular_symmetry_functions,
        start=acsf.num_radial_symmetry_functions,
    ):
        result = result.at[index].set(
            _calculate_angular_acsf_per_atom(
                symmetry_function,
                structure.atom_types,
                position_differences_i,
                distances_i,
                structure.lattice,
                structure.element_map[assigned_elements.neighbor_j],
                structure.element_map[assigned_elements.neighbor_k],  # type: ignore
            )
        )
    return result


_calculate_acsf_descriptor = vmap(
    _calculate_acsf_descriptor_per_atom,
    in_axes=(None, 0, None),
)

_jitted_calculate_acsf_descriptor = jit(
    _calculate_acsf_descriptor,
    static_argnums=(0,),
)

_calculate_grad_acsf_descriptor_per_atom = jacfwd(
    _calculate_acsf_descriptor_per_atom,
    argnums=1,
)

_calculate_grad_acsf_descriptor = vmap(
    _calculate_grad_acsf_descriptor_per_atom,
    in_axes=(None, 0, None),
)

_jitted_calculate_grad_acsf_descriptor = jit(
    _calculate_grad_acsf_descriptor,
    static_argnums=(0,),
)


def _calculate_radial_acsf_per_atom(
    radial_symmetry_function: RadialSymmetryFunction,
    distances_i: Array,
    atom_types: Array,
    neighbor_atom_type_j: Array,
) -> Array:
    r_cutoff = jnp.array(radial_symmetry_function.r_cutoff)
    cutoff_mask_i = _calculate_cutoff_masks_per_atom(distances_i, r_cutoff)
    cutoff_masks_and_atom_types_ij = cutoff_mask_i & (
        atom_types == neighbor_atom_type_j
    )
    return jnp.sum(
        radial_symmetry_function(distances_i),
        where=cutoff_masks_and_atom_types_ij,
        axis=0,
    )


def _calculate_angular_acsf_per_atom(
    angular_symmetry_function: AngularSymmetryFunction,
    atom_types: Array,
    position_differences_i: Array,
    distances_i: Array,
    lattice: Array,
    neighbor_atom_type_j: Array,
    neighbor_atom_type_k: Array,
) -> Array:

    # cutoff-radius masks
    r_cutoff = jnp.array(angular_symmetry_function.r_cutoff)
    cutoff_masks_i = _calculate_cutoff_masks_per_atom(distances_i, r_cutoff)
    # masks for neighboring element j
    cutoff_masks_and_atom_types_ij = cutoff_masks_i & (
        atom_types == neighbor_atom_type_j
    )
    # masks for neighboring element k
    cutoff_masks_and_atom_types_ik = cutoff_masks_i & (
        atom_types == neighbor_atom_type_k
    )
    # angular terms
    total, _ = lax.scan(
        partial(
            _inner_loop_over_angular_acsf_terms,
            mask_ik=cutoff_masks_and_atom_types_ik,
            position_differences_i=position_differences_i,
            distances_i=distances_i,
            lattice=lattice,
            kernel=angular_symmetry_function,
        ),
        jnp.array(0.0),
        (position_differences_i, distances_i, cutoff_masks_and_atom_types_ij),
    )
    # correct the double-counting
    # avoided jit recompilation due to lambda's variable hash
    return lax.cond(
        neighbor_atom_type_j == neighbor_atom_type_k,
        _divide_by_two,
        _return_input,
        total,
    )  # type: ignore


# Called by lax.scan (no need for @jax.jit)
def _inner_loop_over_angular_acsf_terms(
    total: Array,
    inputs: Tuple[Array, ...],
    position_differences_i: Array,
    distances_i: Array,
    mask_ik: Array,
    lattice: Array,
    kernel: Callable[[Array, Array, Array, Array], Array],
) -> Tuple[Array, Array]:
    # Scan occurs along the leading axis
    Rij, rij, mask_ij = inputs
    # fix nan issue in gradient
    # see https://github.com/google/jax/issues/1052#issuecomment-514083352
    operand = rij * distances_i
    is_zero = operand == 0.0
    true_op = jnp.where(is_zero, 1.0, operand)
    cost = jnp.where(
        mask_ik,
        jnp.inner(Rij, position_differences_i) / true_op,
        1.0,
    )
    cost = jnp.where(is_zero, 0.0, cost)  # type: ignore
    rjk = jnp.where(  # diff_jk = diff_ji - position_differences_ik
        mask_ik,
        _calculate_distances_per_atom(Rij, position_differences_i, lattice),
        0.0,
    )  # second tuple output for Rjk
    value = jnp.where(
        mask_ij,
        jnp.sum(
            kernel(rij, distances_i, rjk, cost),
            where=mask_ik & (rjk > 0.0),  # exclude k=j # type:ignore
            axis=0,
        ),
        0.0,
    )
    return total + value, value


def _return_input(array: Array) -> Array:
    return array


def _divide_by_two(array: Array) -> Array:
    return array * 0.5


ACSF = AtomCenteredSymmetryFunction

register_jax_pytree_node(AtomCenteredSymmetryFunction)
