import itertools
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap

from pantea.atoms.distance import (
    _calculate_distances_per_atom,
    _calculate_distances_with_aux_per_atom,
)
from pantea.atoms.neighbor import _calculate_masks_per_atom
from pantea.atoms.structure import Structure
from pantea.descriptors.acsf.angular import AngularSymmetryFunction
from pantea.descriptors.acsf.radial import RadialSymmetryFunction
from pantea.descriptors.acsf.symmetry import BaseSymmetryFunction, EnvironmentElements
from pantea.descriptors.descriptor import DescriptorInterface
from pantea.logger import logger
from pantea.pytree import BaseJaxPytreeDataClass, register_jax_pytree_node
from pantea.types import Array, Element


@jit
def _calculate_radial_acsf_per_atom(
    radial: Dict[EnvironmentElements, RadialSymmetryFunction],
    atype: Array,
    dist_i: Array,
    emap: Dict[Element, Array],
) -> Array:
    elements: EnvironmentElements = [k for k in radial.keys()][0]

    mask_cutoff_i = _calculate_masks_per_atom(
        dist_i, jnp.array(radial[elements].r_cutoff)
    )
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
    emap: Dict[Element, Array],
) -> Array:
    elements: EnvironmentElements = [k for k in angular.keys()][0]

    # cutoff-radius mask
    mask_cutoff_i = _calculate_masks_per_atom(
        dist_i, jnp.array(angular[elements].r_cutoff)
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
        jnp.array(0.0),
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
    cost = jnp.where(is_zero, 0.0, cost)  # type: ignore

    rjk = jnp.where(  # diff_jk = diff_ji - diff_ik
        mask_ik,
        _calculate_distances_per_atom(Rij, diff_i, lattice),
        0.0,
    )  # second tuple output for Rjk

    value = jnp.where(
        mask_ij,
        jnp.sum(
            kernel(rij, dist_i, rjk, cost),
            where=mask_ik & (rjk > 0.0),  # exclude k=j # type:ignore
            axis=0,
        ),
        0.0,
    )

    return total + value, value


class AcsfInterface(Protocol):
    num_symmetry_functions: int
    num_radial_symmetry_functions: int
    num_angular_symmetry_functions: int
    radial_symmetry_functions: Tuple
    angular_symmetry_functions: Tuple


@jit
def _calculate_descriptor_per_atom(
    acsf: AcsfInterface,
    single_atom_position: Array,
    neighbor_positions: Array,
    atype: Array,
    lattice: Array,
    emap: Dict,
) -> Array:
    """
    Compute descriptor values per atom in the structure (via atom id).
    """
    dtype = single_atom_position.dtype
    result: Array = jnp.empty(acsf.num_symmetry_functions, dtype=dtype)

    dist_i, diff_i = _calculate_distances_with_aux_per_atom(
        single_atom_position, neighbor_positions, lattice
    )

    # Loop over the radial terms
    for index, (elements, radial) in enumerate(acsf.radial_symmetry_functions):
        result = result.at[index].set(
            _calculate_radial_acsf_per_atom({elements: radial}, atype, dist_i, emap)
        )

    # Loop over the angular terms
    for index, (elements, angular) in enumerate(
        acsf.angular_symmetry_functions,
        start=acsf.num_radial_symmetry_functions,
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


_calculate_grad_descriptor_per_atom = jax.jacfwd(
    _calculate_descriptor_per_atom,
    argnums=1,
)

_jitted_calculate_grad_descriptor_per_atom = jit(
    _calculate_grad_descriptor_per_atom,
    static_argnums=(0,),
)

_calculate_grad_descriptor_per_element = vmap(
    _calculate_grad_descriptor_per_atom,
    in_axes=(None, 0, None, None, None, None),
)

_jitted_calculate_grad_descriptor_per_element = jit(
    _calculate_grad_descriptor_per_element,
    static_argnums=(0,),
)


@dataclass
class ACSF(BaseJaxPytreeDataClass, DescriptorInterface):
    """
    Atom-centered Symmetry Function (`ACSF`_) descriptor captures
    information about the distribution of neighboring
    atoms around a central atom by considering both radial (two-body) and angular
    (three-body) symmetry functions. Radial symmetry functions describe the distances
    between the central atom and its neighbors, while angular symmetry functions capture
    the angles formed by the central atom with pairs of neighboring atoms.

    The ACSF descriptor is computed by summing the contributions of the `symmetry functions`
    for all neighboring atoms within a specified cutoff distance. The values obtained from
    these calculations represent a fingerprint of the local atomic environment and can
    be used in various machine learning potentials.


    Example
    -------

    .. code-block:: python
        :linenos:

        from pantea.descriptors.acsf import ACSF, G2, G3, G9, CutoffFunction

        # Create an instance of ACSF for `O` element
        acsf = ACSF('O')

        # Set cutoff function and symmetry functions
        cfn = CutoffFunction(12.0)
        g2 = G2(cfn, eta=0.5, r_shift=0.0)
        g3 = G3(cfn, eta=0.001, zeta=2.0, lambda0=1.0, r_shift=12.0)

        # Add different symmetry functions
        acsf.add(g2, 'O')
        acsf.add(g2, 'H')
        acsf.add(g3, 'H', 'H')
        acsf.add(g3, 'H', 'O')

        print(acsf)

    Output:

    .. code-block:: bash

        ACSF(central_element='O', num_symmetry_functions=4, r_cutoff=12.0)


    .. _ACSF: https://compphysvienna.github.io/n2p2/topics/descriptors.html?highlight=symmetry%20function#

    """

    central_element: str
    radial_symmetry_functions: Tuple[Tuple[EnvironmentElements, RadialSymmetryFunction]] = tuple()  # type: ignore
    angular_symmetry_functions: Tuple[Tuple[EnvironmentElements, AngularSymmetryFunction]] = tuple()  # type: ignore

    def add(
        self,
        symmetry_function: BaseSymmetryFunction,
        neighbor_element_j: Element,
        neighbor_element_k: Optional[Element] = None,
    ) -> None:
        """Add the input symmetry function to the list of ACSFs."""
        if isinstance(symmetry_function, RadialSymmetryFunction):
            self.radial_symmetry_functions = self.radial_symmetry_functions + (
                (
                    EnvironmentElements(self.central_element, neighbor_element_j),
                    symmetry_function,
                ),
            )  # type: ignore
        elif isinstance(symmetry_function, AngularSymmetryFunction):
            self.angular_symmetry_functions = self.angular_symmetry_functions + (
                (
                    EnvironmentElements(
                        self.central_element, neighbor_element_j, neighbor_element_k  # type: ignore
                    ),
                    symmetry_function,
                ),
            )
        else:
            logger.error(
                f"Unknown symmetry function type {symmetry_function}",
                exception=TypeError,
            )

    def __call__(
        self,
        structure: Structure,
        atom_indices: Optional[Array] = None,
    ) -> Array:
        """
        Calculate descriptor values for the input given structure and atom index.

        :param structure: input structure instance
        :type structure: Structure
        :param atom_indices: atom indices, defaults select all atom indices
        of type the central element of the descriptor.
        :type atom_indices: Optional[Array], optional
        :return: descriptor values
        :rtype: Array
        """

        if self.num_symmetry_functions == 0:
            logger.warning("No symmetry function was found")

        if atom_indices is None:
            atom_indices = structure.select(self.central_element)
        else:
            atom_indices = jnp.atleast_1d(atom_indices)  # type: ignore
            # Check aid atom type match the central element
            if not jnp.all(
                structure.element_map.element_to_atom_type[self.central_element]
                == structure.atom_types[atom_indices]
            ):
                logger.error(
                    f"Inconsistent central element '{self.central_element}': "
                    f" input atom indices={atom_indices}"
                    f" (atom_types='{int(structure.atom_types[atom_indices])}')",
                    exception=ValueError,
                )

        return _calculate_descriptor(
            self,
            structure.positions[atom_indices],
            structure.positions,
            structure.atom_types,
            structure.lattice,
            structure.element_map.element_to_atom_type,
        )

    def grad_per_element(
        self,
        structure: Structure,
        element: Element,
    ) -> Array:
        """
        Compute gradient of ACSF descriptor respect to the atom position for element.

        :param structure: input Structure instance
        :param element: element exists in the structure
        :return: gradient of the descriptor value respect to the atom position
        """
        element_aids = structure.select(element)
        return _jitted_calculate_grad_descriptor_per_element(
            self,
            structure.positions[element_aids],  # must be element positions shape=(1, 3)
            structure.positions,
            structure.atom_types,
            structure.lattice,
            structure.element_map.element_to_atom_type,
        )

    def grad(
        self,
        structure: Structure,
        atom_index: int,
    ) -> Array:
        """
        Compute gradient of ACSF descriptor respect to the atom position for a single atom.

        :param structure: input Structure instance
        :param atom_index: atom index in Structure [0, natoms)
        :return: gradient of the descriptor value respect to the atom position

        Please note that `grad_per_element` method is way much faster than
        the current implementation of this method method.
        """
        if not (0 <= atom_index < structure.natoms):
            logger.error(
                f"Unexpected {atom_index=}."
                f" The index must be between [0, {structure.natoms})",
                ValueError,
            )

        return _jitted_calculate_grad_descriptor_per_atom(
            self,
            structure.positions[
                atom_index
            ],  # must be a single atom position shape=(1, 3)
            structure.positions,
            structure.atom_types,
            structure.lattice,
            structure.element_map.element_to_atom_type,
        )

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
    def num_descriptors(self) -> int:
        return self.num_symmetry_functions

    @property
    def r_cutoff(self) -> float:  # type: ignore
        """Return the maximum cutoff radius for list of the symmetry functions."""
        return max(
            [
                cfn.r_cutoff
                for (_, cfn) in itertools.chain(
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
            f", symmetry_functions={self.num_symmetry_functions})"
        )


register_jax_pytree_node(ACSF)
