from ..logger import logger
from ..base import _Base
import jax
import jax.numpy as jnp
from functools import partial

Tensor = jnp.ndarray


@jax.jit
def _calculate_cutoff_mask_per_atom(
    aid: Tensor,
    rij: Tensor,
    r_cutoff: Tensor,
) -> Tensor:
    # Select atoms inside cutoff radius
    mask = rij <= r_cutoff
    # exclude self-counting
    mask = mask.at[aid].set(False)
    return mask


_vmap_calculate_neighbor_mask = jax.vmap(
    _calculate_cutoff_mask_per_atom,
    in_axes=(0, 0, None),
)


@partial(jax.jit, static_argnums=(0,))  # FIXME
def _calculate_cutoff_mask(
    structure,
    r_cutoff: Tensor,
) -> Tensor:
    # Tensors no need to be differentiable here
    aids = jnp.arange(structure.natoms)  # all atoms
    rij, _ = structure.calculate_distance(aids)
    return _vmap_calculate_neighbor_mask(aids, rij, r_cutoff)


class Neighbor(_Base):
    """
    Neighbor creates a neighbor list of atoms for an input structure.
    Neighbor class is utilized as a tensor buffer which is responsible for preparing a list
    of neighbor it before using.

    Its design should be independent of the input structure.
    For MD simulations, re-neighboring the list is required every few steps (e.g. by defining a skin radius).
    """

    def __init__(self, r_cutoff: float = None) -> None:
        """
        Initialize the neighbor list.

        :param r_cutoff: Cutoff radius, defaults to None.
        :type r_cutoff: float, optional
        """
        self.r_cutoff = r_cutoff
        self.r_cutoff_updated = False
        self.tensors: Dict[str, Tensor] = None
        super().__init__()

    def set_cutoff_radius(self, r_cutoff: float) -> None:
        """
        :param r_cutoff: A new cutoff radius
        :type r_cutoff: float
        """
        logger.debug(
            f"Resetting Neighbor cutoff radius from {self.r_cutoff} to {r_cutoff}"
        )
        self.r_cutoff = r_cutoff
        self.r_cutoff_updated = True

    # def _init_tensors(self, structure) -> None:
    #     """
    #     Allocating tensors for the neighbor list from the input structure.

    #     :param structure: An instance of Structure
    #     :type structure: Dict[Tensor]
    #     """
    #     # FIXME: reduce natoms*natoms tensor size!
    #     # TODO: define max_num_neighbor to avoid extra memory allocation!

    #     # Avoid re-allocating structure with the same size
    #     try:
    #         if structure.natoms == len(self.number):
    #             logger.debug("Avoid re-allocating the neighbor tensors")
    #             return
    #     except AttributeError:
    #         pass

    #     logger.debug("Allocating tensors for neighbor list")
    #     self.tensors = {
    #         "number": jnp.empty(
    #             structure.natoms,
    #             dtype=int,  # dtype.UINT, FIXME
    #         ),
    #         "index": jnp.empty(
    #             (structure.natoms, structure.natoms),
    #             dtype=int,  # dtype.INDEX, FIXME
    #         ),
    #     }

    #     for attr, tensor in self.tensors.items():
    #         logger.debug(
    #             f"{attr:12} -> Tensor(shape='{tensor.shape}', dtype='{tensor.dtype}'"
    #         )
    #     set_as_attribute(self, self.tensors)

    def update(self, structure) -> None:
        """
        This method updates the neighbor atom tensors including the number of neighbor and neighbor atom
        indices for the input structure.
        """
        # TODO: optimize updating the neighbor list, for example using the cell mesh, bin atoms (miniMD), etc.
        if not self.r_cutoff:
            logger.debug("Skipped updating the neighbor list (no cutoff radius)")
            return

        if not structure.requires_neighbor_update and not self.r_cutoff_updated:
            logger.debug("Skipped updating the neighbor list")
            return

        # self._init_tensors(structure) # FIXME: remove

        # ----------------------------------------
        logger.debug("Updating neighbor list")

        self.mask = _calculate_cutoff_mask(
            structure,
            jnp.atleast_1d(self.r_cutoff),
        )

        # Avoid updating the neighbor list the next time
        structure.requires_neighbor_update = False
        self.r_cutoff_updated = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r_cutoff={self.r_cutoff})"
