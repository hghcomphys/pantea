from ...logger import logger
from ...structure import Structure
from ..base import Descriptor
from .symmetry import SymmetryFunction
from .angular import AngularSymmetryFunction
from .radial import RadialSymmetryFunction
from ._asf import _calculate_descriptor
from typing import Tuple, List, Optional
import jax
import itertools
import jax.numpy as jnp


class AtomicSymmetryFunction(Descriptor):
    """
    Atomic Symmetry Function (ASF) descriptor.
    ASF is a vector of different radial and angular terms which describe the chemical environment of an atom.
    """

    # TODO: ASF should be independent of the input structure, but it should knows how to calculate the descriptor vector.
    # See N2P2 -> https://compphysvienna.github.io/n2p2/topics/descriptors.html?highlight=symmetry%20function#

    def __init__(self, element: str) -> None:
        super().__init__(element)  # central element
        self._radial: List[Tuple[RadialSymmetryFunction, str, str]] = list()
        self._angular: List[Tuple[AngularSymmetryFunction, str, str, str]] = list()
        logger.debug(f"Initializing {self}")

    def add(
        self,
        symmetry_function: SymmetryFunction,
        neighbor_element_j: str,
        neighbor_element_k: Optional[str] = None,
    ) -> None:
        """
        This method adds an input symmetry function to the list of ASFs
        and assign it to the given neighbor element(s).
        """
        if isinstance(symmetry_function, RadialSymmetryFunction):
            self._radial.append(
                (
                    symmetry_function,
                    self.element,
                    neighbor_element_j,
                )
            )
        elif isinstance(symmetry_function, AngularSymmetryFunction):
            self._angular.append(
                (
                    symmetry_function,
                    self.element,
                    neighbor_element_j,
                    neighbor_element_k,
                )
            )
        else:
            logger.error(
                f"Unknown input symmetry function type {symmetry_function}",
                exception=TypeError,
            )

    def __call__(
        self,
        structure: Structure,
        aid: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Calculate descriptor values for the input given structure and atom id(s).
        """
        # Check number of symmetry functions
        if self.n_symmetry_functions == 0:
            logger.warning(
                f"No symmetry function defined yet:"
                f" radial={self.n_radial_symmetry_functions}"
                f", angular={self.n_angular_symmetry_functions}"
            )

        if aid is None:
            aid = structure.select(self.element)
        else:
            aid = jnp.atleast_1d(aid)
            # Check aid atom type match the central element
            if not jnp.all(
                structure.element_map.element_to_atype[self.element]
                == structure.atype[aid]
            ):
                logger.error(
                    f"Inconsistent central element '{self.element}': input aid={aid}"
                    f" (atype='{int(structure.atype[aid])}')",
                    exception=ValueError,
                )

        return _calculate_descriptor(
            self,
            structure.position[aid],
            structure.position,
            structure.atype,
            structure.box.lattice,
            structure.dtype,
            structure.element_map.element_to_atype,
        )

    def grad(
        self,
        structure: Structure,
        asf_index: int,
        atom_index: int,
    ):
        if not (0 <= asf_index < self.n_symmetry_functions):
            logger.error(
                f"Unexpected {asf_index=}."
                f" The index must be between [0, {self.n_symmetry_functions})",
                ValueError,
            )

        if not (0 <= atom_index < structure.n_atoms):
            logger.error(
                f"Unexpected {asf_index=}."
                f" The index must be between [0, {structure.n_atoms})",
                ValueError,
            )

        def asf_fn(pos):
            return _calculate_descriptor(
                self,
                pos,
                structure.position,
                structure.atype,
                structure.box.lattice,
                structure.dtype,
                structure.element_map.element_to_atype,
            )[0][asf_index]

        # FIXME: error when using vmap on grad over aids
        # TODO: add jit
        grad_asf_fn = jax.grad(asf_fn)
        pos = structure.position[atom_index]

        return grad_asf_fn(pos[None, :])

    # def grad2(
    #     self,
    #     structure: Structure,
    #     asf_index: int,
    #     aid: Optional[jnp.ndarray] = None,
    # ):
    #     if asf_index > self.n_symmetry_functions - 1:
    #         logger.error(
    #             f"Unexpected ASF array index {asf_index}."
    #             f" The index must be between [0, {self.n_symmetry_functions})",
    #             ValueError,
    #         )

    #     if aid is None:
    #         aid = jnp.arange(structure.n_atoms)

    #     return _vmap_grad_func_asf(
    #         self,
    #         asf_index,
    #         jnp.atleast_1d(aid),
    #         structure.position,
    #         structure.box.lattice,
    #         structure.atype,
    #         structure.element_map.element_to_atype,
    #     )

    @property
    def n_radial_symmetry_functions(self) -> int:
        return len(self._radial)

    @property
    def n_angular_symmetry_functions(self) -> int:
        return len(self._angular)

    @property
    def n_symmetry_functions(self) -> int:
        return self.n_radial_symmetry_functions + self.n_angular_symmetry_functions

    @property
    def r_cutoff(self) -> float:
        """
        Return the maximum cutoff radius of all descriptor terms.
        """
        return max(
            [cfn[0].r_cutoff for cfn in itertools.chain(*[self._radial, self._angular])]
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(element='{self.element}'"
            f", radial={self.n_radial_symmetry_functions}"
            f", angular={self.n_angular_symmetry_functions})"
        )


# Define ASF alias
ASF = AtomicSymmetryFunction
