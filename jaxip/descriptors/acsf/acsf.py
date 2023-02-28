import itertools
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from jaxip.base import register_jax_pytree_node
from jaxip.descriptors.acsf._acsf import (
    _calculate_descriptor,
    _calculate_grad_descriptor_per_atom,
)
from jaxip.descriptors.acsf.angular import AngularSymmetryFunction
from jaxip.descriptors.acsf.radial import RadialSymmetryFunction
from jaxip.descriptors.acsf.symmetry import EnvironmentElements, SymmetryFunction
from jaxip.descriptors.base import Descriptor
from jaxip.logger import logger
from jaxip.structure.structure import Structure
from jaxip.types import Array, Element


@dataclass
class ACSF(Descriptor):
    """
    Atom-centered Symmetry Function (`ACSF`_) descriptor.

    ACSF describes the local environment of an atom (neighbors' distribution).
    It usually contains multiple combinations of `radial` (two-body)
    and `angular` (three-body) symmetry functions.

    .. note::
        The ACSF is independent of input structure
        but it knows how to calculate the descriptor values for any given structures.

    Example
    -------

    .. code-block:: python
        :linenos:

        from jaxip.descriptors.acsf import ACSF, G2, G3, G9, CutoffFunction

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

    This gives the following output:

    .. code-block:: bash

        ACSF(element='O', num_symmetry_functions=4, r_cutoff=12.0)

    .. _ACSF: https://compphysvienna.github.io/n2p2/topics/descriptors.html?highlight=symmetry%20function#

    """

    element: str
    # Here we use hashable tuple instead of list due to JIT compilation
    # FIXME: set radial and angular attribute as dynamic JIT arguments
    radial_symmetry_functions: Tuple[Tuple[EnvironmentElements, RadialSymmetryFunction]] = tuple()  # type: ignore
    angular_symmetry_functions: Tuple[Tuple[EnvironmentElements, AngularSymmetryFunction]] = tuple()  # type: ignore

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    def add(
        self,
        symmetry_function: SymmetryFunction,
        neighbor_element_j: Element,
        neighbor_element_k: Optional[Element] = None,
    ) -> None:
        """Add the input symmetry function to the list of ACSFs."""
        if isinstance(symmetry_function, RadialSymmetryFunction):
            self.radial_symmetry_functions = self.radial_symmetry_functions + (
                (
                    EnvironmentElements(self.element, neighbor_element_j),
                    symmetry_function,
                ),
            )  # type: ignore
        elif isinstance(symmetry_function, AngularSymmetryFunction):
            self.angular_symmetry_functions = self.angular_symmetry_functions + (
                (
                    EnvironmentElements(
                        self.element, neighbor_element_j, neighbor_element_k  # type: ignore
                    ),
                    symmetry_function,
                ),
            )
        else:
            logger.error(
                f"Unknown input symmetry function type {symmetry_function}",
                exception=TypeError,
            )

    def __call__(
        self,
        structure: Structure,
        atom_index: Optional[Array] = None,
    ) -> Array:
        """
        Calculate descriptor values for the input given structure and atom index.

        :param structure: input structure instance
        :type structure: Structure
        :param atom_index: index for atom(s), defaults select all atom indices of type the central element of the descriptor.
        :type atom_index: Optional[Array], optional
        :return: descriptor values
        :rtype: Array
        """

        if self.num_symmetry_functions == 0:
            logger.warning(
                f"No symmetry function defined yet:"
                f" radial={self.num_radial_symmetry_functions}"
                f", angular={self.num_angular_symmetry_functions}"
            )

        if atom_index is None:
            atom_index = structure.select(self.element)
        else:
            atom_index = jnp.atleast_1d(atom_index)
            # Check aid atom type match the central element
            if not jnp.all(
                structure.element_map.element_to_atype[self.element]
                == structure.atom_type[atom_index]
            ):
                logger.error(
                    f"Inconsistent central element '{self.element}': input aid={atom_index}"
                    f" (atype='{int(structure.atom_type[atom_index])}')",
                    exception=ValueError,
                )

        return _calculate_descriptor(
            self,
            structure.position[atom_index],
            structure.position,
            structure.atom_type,
            structure.box.lattice,
            structure.element_map.element_to_atype,
        )

    def grad(
        self,
        structure: Structure,
        atom_index: int,
    ) -> Array:
        """
        Compute gradient of ACSF descriptor respect to atom position (x, y, and z).

        :param structure: input structure instance
        :type structure: Structure
        :param atom_index: atom index in the structure [0, natoms)
        :type atom_index: int
        :return: gradient of the descriptor value respect to the atom position
        :rtype: Array
        """
        if not (0 <= atom_index < structure.natoms):
            logger.error(
                f"Unexpected {atom_index=}."
                f" The index must be between [0, {structure.natoms})",
                ValueError,
            )

        # TODO: extend it to multiple atom indices
        return _calculate_grad_descriptor_per_atom(
            self,
            structure.position[
                atom_index
            ],  # must be a single atom position shape=(1, 3)
            structure.position,
            structure.atom_type,
            structure.box.lattice,
            structure.element_map.element_to_atype,
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
                    *[self.radial_symmetry_functions, self.angular_symmetry_functions]
                )
            ]
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(element='{self.element}'"
            f", num_symmetry_functions={self.num_symmetry_functions}"
            f", r_cutoff={self.r_cutoff})"
        )


register_jax_pytree_node(ACSF)
