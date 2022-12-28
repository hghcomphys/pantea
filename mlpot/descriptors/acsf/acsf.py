import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp

from mlpot.base import register_jax_pytree_node
from mlpot.descriptors.acsf._acsf import _calculate_descriptor
from mlpot.descriptors.acsf.angular import AngularElements, AngularSymmetryFunction
from mlpot.descriptors.acsf.radial import RadialElements, RadialSymmetryFunction
from mlpot.descriptors.acsf.symmetry import SymmetryFunction
from mlpot.descriptors.base import Descriptor
from mlpot.logger import logger
from mlpot.structure.structure import Structure
from mlpot.types import Array


@dataclass
class ACSF(Descriptor):
    """
    Atom-centered Symmetry Function (`ACSF`_) descriptor.

    ACSF describes the chemical environment of an atom.
    It composes of `two-body` (radial) and `three-body` (angular) symmetry functions.

    .. note::
        The ACSF is independent of the input structure,
        but it knows how to calculate the descriptor values for any given structure.

    .. _ACSF: https://compphysvienna.github.io/n2p2/topics/descriptors.html?highlight=symmetry%20function#
    """

    element: str
    radial: Tuple[Tuple[RadialElements, RadialSymmetryFunction]] = tuple()
    angular: Tuple[Tuple[AngularElements, AngularSymmetryFunction]] = tuple()

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    def add(
        self,
        symmetry_function: SymmetryFunction,
        neighbor_element_j: str,
        neighbor_element_k: Optional[str] = None,
    ) -> None:
        """Add the input symmetry function to the list of ACSFs."""
        if isinstance(symmetry_function, RadialSymmetryFunction):
            self.radial = self.radial + (
                (
                    RadialElements(self.element, neighbor_element_j),
                    symmetry_function,
                ),
            )  # type: ignore
        elif isinstance(symmetry_function, AngularSymmetryFunction):
            self.angular = self.angular + (
                (
                    AngularElements(
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
        asf_index: int,
        atom_index: int,
    ) -> Array:
        """
        Compute gradient of ACSF descriptor respect to atom position (x, y, and z).

        :param structure: input structure instance
        :type structure: Structure
        :param asf_index: index of array in descriptor array, [0, `num_of_symmetry_functions`]
        :type asf_index: int
        :param atom_index: between [0, natoms)
        :type atom_index: int
        :return: gradient respect to position
        :rtype: Array
        """
        if not (0 <= asf_index < self.num_symmetry_functions):
            logger.error(
                f"Unexpected {asf_index=}."
                f" The index must be between [0, {self.num_symmetry_functions})",
                ValueError,
            )

        if not (0 <= atom_index < structure.natoms):
            logger.error(
                f"Unexpected {asf_index=}."
                f" The index must be between [0, {structure.natoms})",
                ValueError,
            )

        def asf_func(pos) -> Array:
            return _calculate_descriptor(
                self,
                pos,
                structure.position,
                structure.atom_type,
                structure.box.lattice,
                structure.element_map.element_to_atype,
            )[0][asf_index]

        # FIXME: error when using vmap on grad over aids
        # TODO: add jit
        grad_asf_fn = jax.grad(asf_func)
        pos: Array = structure.position[atom_index]

        return grad_asf_fn(pos[None, :])

    @property
    def num_radial_symmetry_functions(self) -> int:
        """Return number of `two-body` (radial) symmetry functions."""
        return len(self.radial)

    @property
    def num_angular_symmetry_functions(self) -> int:
        """Return number of `three-body` (angular) symmetry functions."""
        return len(self.angular)

    @property
    def num_symmetry_functions(self) -> int:
        """Return the total (`two-body` and `tree-body`) number of symmetry functions."""
        return self.num_radial_symmetry_functions + self.num_angular_symmetry_functions

    @property
    def r_cutoff(self) -> float:  # type: ignore
        """Return the maximum cutoff radius for list of the symmetry functions."""
        return max(
            [cfn[0].r_cutoff for cfn in itertools.chain(*[self.radial, self.angular])]
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(element='{self.element}'"
            f", num_radial={self.num_radial_symmetry_functions}"
            f", num_angular={self.num_angular_symmetry_functions})"
        )


register_jax_pytree_node(ACSF)
