from torchip.descriptors.asf.symmetry import SymmetryFunction
from ...logger import logger
from ...structure import Structure
from ...structure import _calculate_cutoff_mask_per_atom
from ..base import Descriptor
from .angular import AngularSymmetryFunction
from .radial import RadialSymmetryFunction
from typing import Tuple, Union, Optional
import itertools
import jax
import jax.numpy as jnp
from functools import partial

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
        self._radial: Tuple[RadialSymmetryFunction, str, str] = list()
        self._angular: Tuple[AngularSymmetryFunction, str, str, str] = list()
        # self.__cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-8) # instantiate
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
        structure.update_neighbor()  # TODO: remove?

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

        # Return descriptor values
        # return jax.vmap(
        #     self._calculate_descriptor_per_atom,
        #     in_axes=(None, None, 0),
        # )(structure, aid)
        # return self._calculate_descriptor_per_atom(structure, aid)
        return jnp.stack(
            [self._calculate_descriptor_per_atom(structure, aid_) for aid_ in aid]
        )  # FIXME: vmap

    @partial(jax.jit, static_argnums=(0, 1))  # FIXME
    def _calculate_descriptor_per_atom(
        self,
        structure: Structure,
        aid: Tensor,
    ) -> Tensor:
        """
        [Kernel]
        Compute descriptor values of an input atom id for the given structure tensors.
        """
        # pos = structure.position
        # ms = structure.neighbor.mask
        # lattice = structure.box.lattice
        atype = structure.atype
        emap = structure.element_map.element_to_atype

        # A tensor for final descriptor values of a single atom
        result = jnp.zeros(self.n_descriptor, dtype=structure.dtype)

        # Get cutoff mask for the atom
        # ms_ = jnp.squeeze(ms[aid])
        # Calculate distances of neighboring atoms
        dis_i, diff_i = structure.calculate_distance(
            aid
        )  # TODO: use _calculate_distance
        # Get the corresponding neighboring atom types
        # at_ refers to the array atom type of neighbors
        # at_ = jnp.where(ms_, at, -1)

        # Loop over the radial terms
        for radial_index, radial in enumerate(self._radial):

            # Find neighboring atom indices that match the given ASF cutoff radius AND atom type
            mask_cutoff_i = _calculate_cutoff_mask_per_atom(
                aid, dis_i, jnp.atleast_1d(radial[0].r_cutoff)
            )
            mask_cutoff_and_atype_ij = mask_cutoff_i & (atype == emap[radial[2]])

            # Apply radial ASF term kernels and sum over the all neighboring atoms
            result = result.at[radial_index].set(
                jnp.sum(
                    radial[0](dis_i),
                    where=mask_cutoff_and_atype_ij,
                    axis=0,
                )
            )

        # Loop over the angular terms
        for angular_index, angular in enumerate(self._angular, start=self.n_radial):

            # Find neighboring atom indices that match the given ASF cutoff radius
            mask_cutoff_i = _calculate_cutoff_mask_per_atom(
                aid, dis_i, jnp.atleast_1d(angular[0].r_cutoff)
            )

            # Find LOCAL indices of neighboring elements j and k
            at_j, at_k = emap[angular[2]], emap[angular[3]]
            mask_cutoff_and_atype_ij = mask_cutoff_i & (atype == at_j)
            mask_cutoff_and_atype_ik = mask_cutoff_i & (atype == at_k)

            # Apply angular ASF term kernels and sum over the neighboring atoms
            total = jnp.asarray(0.0)
            for Rij, rij, mask in zip(diff_i, dis_i, mask_cutoff_and_atype_ij):

                rjk = rij  # FIXME
                # mask_cutoff_jk = _

                rik = jnp.where(
                    mask_cutoff_and_atype_ik,
                    dis_i,
                    0.0,
                )

                cost = jnp.where(
                    mask_cutoff_and_atype_ik,
                    jnp.inner(Rij, diff_i) / (rij * dis_i),
                    0.0,
                )

                total += jnp.where(
                    mask,
                    jnp.sum(
                        angular[0](rij, rik, rjk, cost),
                        where=mask_cutoff_and_atype_ik,
                    ),
                    0.0,
                )

            result = result.at[angular_index].set(total)

            # # loop over neighbor element 1 (j)
            # for j in ni_rc_at_j_:
            #     # ----
            #     ni_j_ = ni_[j]  # neighbor atom index for j (scalar)
            #     # k = ni_rc_at_k_[ ni_[ni_rc_at_k_] > ni_j_ ]
            #     # # apply k > j (k,j != i is already applied in the neighbor list)
            #     if at_j == at_k:  # TODO: why? dedicated k and j list to each element
            #         k = ni_rc_at_k_[ni_[ni_rc_at_k_] > ni_j_]
            #     else:
            #         k = ni_rc_at_k_[ni_[ni_rc_at_k_] != ni_j_]
            #     ni_k_ = ni_[k]  # neighbor atom index for k (an array)

            #     # ---
            #     rij = dis_[j]  # shape=(1), LOCAL index j
            #     rik = dis_[k]  # shape=(*), LOCAL index k (an array)
            #     Rij = diff_[j]  # x[aid] - x[ni_j_] # shape=(3)
            #     Rik = diff_[k]  # x[aid] - x[ni_k_] # shape=(*, 3)

            #     # ---
            #     rjk, _ = _calculate_distance(
            #         pos[ni_j_], pos[ni_k_], lattice=lattice
            #     )  # shape=(*)
            #     # Rjk = structure.apply_pbc(x[ni_j_] - x[ni_k_])   # shape=(*, 3)
            #     # ---
            #     # Cosine of angle between k--<i>--j atoms
            #     # TODO: move cosine calculation to structure
            #     # cost = self.__cosine_similarity(Rij.expand(Rik.shape), Rik)   # shape=(*)
            #     cost = jnp.inner(Rij, Rik) / (rij * rik)
            #     # ---
            #     # Broadcasting computation (avoiding to use the in-place add() because of autograd)
            #     result = result.at[angular_index].add(
            #         jnp.sum(
            #             angular[0](rij, rik, rjk, cost),
            #             axis=0,
            #         )
            #     )

        return result

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
