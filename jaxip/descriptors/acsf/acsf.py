import itertools
from dataclasses import dataclass
from typing import Optional, Tuple

import jax.numpy as jnp

from jaxip.atoms.structure import Structure
from jaxip.descriptors.acsf._acsf import (
    _calculate_descriptor,
    _calculate_grad_descriptor_per_atom,
)
from jaxip.descriptors.acsf.angular import AngularSymmetryFunction
from jaxip.descriptors.acsf.radial import RadialSymmetryFunction
from jaxip.descriptors.acsf.symmetry import (
    BaseSymmetryFunction,
    EnvironmentElements,
)
from jaxip.descriptors.descriptor import DescriptorInterface
from jaxip.logger import logger
from jaxip.pytree import BaseJaxPytreeDataClass, register_jax_pytree_node
from jaxip.types import Array, Element


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
                    EnvironmentElements(
                        self.central_element, neighbor_element_j
                    ),
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
                structure.element_map.element_to_atom_type[
                    self.central_element
                ]
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

    def grad(
        self,
        structure: Structure,
        atom_index: int,
    ) -> Array:
        """
        Compute gradient of ACSF descriptor respect to atom position (x, y, and z).

        :param structure: input Structure instance
        :param atom_index: atom index in Structure [0, natoms)
        :return: gradient of the descriptor value respect to the atom position
        """
        if not (0 <= atom_index < structure.natoms):
            logger.error(
                f"Unexpected {atom_index=}."
                f" The index must be between [0, {structure.natoms})",
                ValueError,
            )

        return _calculate_grad_descriptor_per_atom(
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
        return (
            self.num_radial_symmetry_functions
            + self.num_angular_symmetry_functions
        )

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
