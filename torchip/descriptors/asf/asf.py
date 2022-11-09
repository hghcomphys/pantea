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
    # ) # same for Rjk if using diff_i

    # fix nan issue in gradient
    # see https://github.com/google/jax/issues/1052#issuecomment-514083352
    value = rij * dist_i
    true_op = jnp.where(value == 0.0, 1.0, value)

    cost = jnp.where(
        mask_ik,
        jnp.inner(Rij, diff_i) / true_op,
        1.0,
    )
    rjk = jnp.where(  # diff_jk = diff_ji - diff_ik
        mask_ik,
        _calculate_distance_per_atom(Rij, diff_i, lattice)[0],
        0.0,
    )  # second tuple output for Rjk

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


@partial(jax.jit, static_argnums=(0, 5))  # FIXME
def _calculate_descriptor_per_atom(
    asf,
    aid: Tensor,  # must be a single atom id
    position: Tensor,
    atype: Tensor,
    lattice: Tensor,
    dtype: jnp.dtype,
    emap: Dict,
) -> Tensor:
    """
    Compute descriptor values per atom in the structure (via atom id).
    """
    # returned descriptor array
    result = jnp.zeros(asf.n_descriptor, dtype=dtype)

    dist_i, diff_i = _calculate_distance_per_atom(position[aid], position, lattice)

    # Loop over the radial terms
    for radial_index, radial in enumerate(asf._radial):

        # cutoff-radius mask
        mask_cutoff_i = _calculate_cutoff_mask_per_atom(
            aid, dist_i, jnp.asarray(radial[0].r_cutoff)
        )
        # get the corresponding neighboring atom types
        mask_cutoff_and_atype_ij = mask_cutoff_i & (atype == emap[radial[2]])

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
            aid, dist_i, jnp.asarray(angular[0].r_cutoff)
        )
        # mask for neighboring element j
        at_j = emap[angular[2]]
        mask_cutoff_and_atype_ij = mask_cutoff_i & (atype == at_j)
        # mask for neighboring element k
        at_k = emap[angular[3]]
        mask_cutoff_and_atype_ik = mask_cutoff_i & (atype == at_k)

        total, _ = jax.lax.scan(
            partial(
                _inner_loop_over_angular_terms,
                mask_ik=mask_cutoff_and_atype_ik,
                diff_i=diff_i,
                dist_i=dist_i,
                lattice=lattice,
                kernel=angular[0],
            ),
            0.0,
            (diff_i, dist_i, mask_cutoff_and_atype_ij),
        )
        # correct double-counting
        total = jax.lax.cond(at_j == at_k, lambda x: 0.5 * x, lambda x: x, total)

        result = result.at[angular_index].set(total)

    return result


_calculate_descriptor = jax.vmap(
    _calculate_descriptor_per_atom,
    in_axes=(None, 0, None, None, None, None, None),  # vmap only on aid
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

    def __call__(
        self,
        structure: Structure,
        aid: Optional[Union[int, Tensor]] = None,
    ) -> Tensor:
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

    # @partial(jax.jit, static_argnums=(0,))  # FIXME
    def grad(self, structure, aid=None):

        aid = jnp.atleast_1d(aid)

        # vgrad_calculate_descriptor = jax.vmap(
        #     jax.grad(_calculate_descriptor_per_atom, argnums=3),  # gradient respect to position
        #     in_axes=(None, None, 1, None),  # vmap on aid
        # )
        # return vgrad_calculate_descriptor(self, structure, aid, structure.position)

        def kernel(asf, aid, position, atype, lattice, dtype, emap):
            return _calculate_descriptor_per_atom(
                asf, aid, position, atype, lattice, dtype, emap
            ).sum()

        # gradient respect to position
        grad_calculate_descriptor_per_atom = jax.grad(
            kernel,
            argnums=2,
        )

        return grad_calculate_descriptor_per_atom(
            self,
            aid,
            structure.position,
            structure.atype,
            structure.box.lattice,
            structure.dtype,
            structure.element_map.element_to_atype,
        )

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
