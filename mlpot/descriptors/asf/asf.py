from ...logger import logger
from ...structure import Structure
from ..base import Descriptor
from .symmetry import SymmetryFunction
from .angular import AngularSymmetryFunction
from .radial import RadialSymmetryFunction
from ._asf import _calculate_descriptor
from ._grad import _grad_asf_func_radial
from ._grad import _grad_asf_func_angular
from typing import Tuple, List, Optional
import itertools
import jax.numpy as jnp


Tensor = jnp.ndarray


class AtomicSymmetryFunction(Descriptor):
    """
    Atomic Symmetry Function (ASF) descriptor.
    ASF is a vector of different radial and angular terms which describe the chemical environment of an atom.
    TODO: ASF should be independent of the input structure, but it should knows how to calculate the descriptor vector.
    See N2P2 -> https://compphysvienna.github.io/n2p2/topics/descriptors.html?highlight=symmetry%20function#
    """

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
        # TODO: tuple of dict? (tuple is fine if it's used internally)
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

    def __call__(self, structure: Structure, aid: Optional[Tensor] = None) -> Tensor:
        """
        Calculate descriptor values for the input given structure and atom id(s).
        """
        # Check number of symmetry functions
        if self.n_descriptor == 0:
            logger.warning(
                f"No symmetry function was found: radial={self.n_radial}"
                f", angular={self.n_angular}"
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
            aid,
            structure.position,
            structure.atype,
            structure.box.lattice,
            structure.dtype,
            structure.element_map.element_to_atype,
        )

    def grad(self, structure: Structure, aid: Tensor, descriptor_index: int):

        # TODO: add logging
        assert descriptor_index < self.n_descriptor

        position = structure.position
        lattice = structure.lattice
        atype = structure.atype
        emap = structure.element_map.element_to_atype

        aid = jnp.atleast_1d(aid)
        if descriptor_index < self.n_radial:
            grad_value = _grad_asf_func_radial(
                self._radial[descriptor_index],
                position[aid],
                aid,
                position,
                lattice,
                atype,
                emap,
            )
        else:
            grad_value = _grad_asf_func_angular(
                self._angular[descriptor_index - self.n_radial],
                position[aid],
                aid,
                position,
                lattice,
                atype,
                emap,
            )
        return jnp.squeeze(grad_value)

    @property
    def n_radial(self) -> int:
        return len(self._radial)

    @property
    def n_angular(self) -> int:
        return len(self._angular)

    @property
    def n_descriptor(self) -> int:
        return self.n_radial + self.n_angular

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
            f"{self.__class__.__name__}(element='{self.element}', n_radial={self.n_radial}"
            f", n_angular={self.n_angular})"
        )


ASF = AtomicSymmetryFunction
