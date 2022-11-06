from torchip.descriptors.asf.symmetry import SymmetryFunction
from ...logger import logger
from ...structure import Structure
from ...structure import _calculate_distance_per_atom
from ...structure import _calculate_cutoff_mask_per_atom
from ..base import Descriptor
from .angular import AngularSymmetryFunction
from .radial import RadialSymmetryFunction
from typing import Callable, Tuple, List, Union, Optional, Dict
import itertools
import jax
import jax.numpy as jnp
from functools import partial

Tensor = jnp.ndarray


# Called by jax.lax.scan (no need for @jax.jit)
def _inner_loop_over_angular_terms(
    total: Tensor,
    inputs: Tuple[Tensor],
    diff_i: Tensor,
    dist_i: Tensor,
    mask_ik: Tensor,
    lattice: Tensor,
    kernel: Callable,
) -> Tuple[Tensor, Tensor]:

    # Scan occurs along the leading axis
    Rij, rij, mask_ij = inputs

    # rik = jnp.where(
    #     mask_cutoff_and_atype_ik,
    #     dist_i,
    #     0.0,
    # ) # same for Rjk
    cost = jnp.where(
        mask_ik,
        jnp.inner(Rij, diff_i) / (rij * dist_i),
        0.0,
    )
    rjk = jnp.where(  # dx_jk = dx_ji - dx_ik
        mask_ik,
        _calculate_distance_per_atom(Rij, diff_i, lattice)[0],
        0.0,
    )  # second output for Rjk

    # TODO: using jax.lax.cond?
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


@partial(jax.jit, static_argnums=(0, 1))  # FIXME
def _calculate_descriptor_per_atom(
    asf,
    structure,
    aid: Tensor,  # expected only one atom id
    # position: Tensor,  # because of grad, must be an explicit input TODO: remove?
) -> Tensor:
    """
    Compute descriptor values per atom in the structure (via atom id).
    """

    # returned descriptor array
    result = jnp.zeros(asf.n_descriptor, dtype=structure.dtype)

    dist_i, diff_i = _calculate_distance_per_atom(
        structure.position[aid], structure.position, structure.lattice
    )

    # Loop over the radial terms
    for radial_index, radial in enumerate(asf._radial):

        # cutoff-radius mask
        mask_cutoff_i = _calculate_cutoff_mask_per_atom(
            aid, dist_i, jnp.atleast_1d(radial[0].r_cutoff)
        )
        # get the corresponding neighboring atom types
        mask_cutoff_and_atype_ij = mask_cutoff_i & (
            structure.atype == structure.element_map.element_to_atype[radial[2]]
        )

        result = result.at[radial_index].set(
            jnp.sum(
                radial[0](dist_i),
                where=mask_cutoff_and_atype_ij,
                axis=0,
            )
        )

    # Loop over the angular terms
    for angular_index, angular in enumerate(asf._angular, start=asf.n_radial):

        # cutoff-radius mask
        mask_cutoff_i = _calculate_cutoff_mask_per_atom(
            aid, dist_i, jnp.atleast_1d(angular[0].r_cutoff)
        )
        # mask for neighboring element j
        at_j = structure.element_map.element_to_atype[angular[2]]
        mask_cutoff_and_atype_ij = mask_cutoff_i & (structure.atype == at_j)
        # mask for neighboring element k
        at_k = structure.element_map.element_to_atype[angular[3]]
        mask_cutoff_and_atype_ik = mask_cutoff_i & (structure.atype == at_k)

        total, _ = jax.lax.scan(
            partial(
                _inner_loop_over_angular_terms,
                mask_ik=mask_cutoff_and_atype_ik,
                diff_i=diff_i,
                dist_i=dist_i,
                lattice=structure.lattice,
                kernel=angular[0],
            ),
            0.0,
            (diff_i, dist_i, mask_cutoff_and_atype_ij),
        )
        # correct double-counting
        if at_j == at_k:
            total *= 0.5

        result = result.at[angular_index].set(total)

    return result


_calculate_descriptor = jax.vmap(
    _calculate_descriptor_per_atom,
    in_axes=(None, None, 0),  # vmap only on aid
)


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

    def register(
        self,
        symmetry_function: SymmetryFunction,
        neighbor_element1: str,
        neighbor_element2: str = None,
    ) -> None:
        """
        This method registers an input symmetry function to the list of ASFs and assign it to the given neighbor element(s).
        # TODO: tuple of dict? (tuple is fine if it's used internally)
        # TODO: solve the confusion for aid, starting from 0 or 1?!
        """
        if isinstance(symmetry_function, RadialSymmetryFunction):
            self._radial.append((symmetry_function, self.element, neighbor_element1))
        elif isinstance(symmetry_function, AngularSymmetryFunction):
            self._angular.append(
                (symmetry_function, self.element, neighbor_element1, neighbor_element2)
            )
        else:
            logger.error("Unknown input symmetry function type", exception=TypeError)

    def __call__(
        self,
        structure: Structure,
        aid: Optional[Union[int, Tensor]] = None,
    ) -> Tensor:
        """
        Calculate descriptor values for the input given structure and atom id(s).
        """
        # Update neighbor list if needed
        # structure.update_neighbor()

        # Check number of symmetry functions
        if self.n_descriptor == 0:
            logger.warning(
                f"No symmetry function was found: radial={self.n_radial}, angular={self.n_angular}"
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
                    f"Inconsistent central element ('{self.element}'): input aid={aid}"
                    f" (atype='{int(structure.atype[aid])}')",
                    exception=ValueError,
                )

        # FIXME: jax.jit
        return _calculate_descriptor(self, structure, aid)

    # # @partial(jax.jit, static_argnums=(0,))  # FIXME
    # def grad(self, structure, aid=None):
    #     aid = jnp.atleast_1d(aid)

    #     vgrad_calculate_descriptor = jax.vmap(
    #         jax.grad(_calculate_descriptor_per_atom, argnums=3),  # gradient respect to position
    #         in_axes=(None, None, 1, None),  # vmap on aid
    #     )
    #     return vgrad_calculate_descriptor(self, structure, aid, structure.position)

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
