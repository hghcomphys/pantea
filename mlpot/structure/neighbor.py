from ..logger import logger
from typing import Optional
from ..base import _Base
from ._neighbor import _calculate_cutoff_mask
import jax.numpy as jnp


class Neighbor(_Base):
    """
    Neighbor creates a neighbor list of atoms for an input structure.
    Neighbor class is utilized as a tensor buffer which is responsible for preparing a list
    of neighbor it before using.

    Its design should be independent of the input structure.
    For MD simulations, re-neighboring the list is required every few steps (e.g. by defining a skin radius).
    """

    def __init__(self, r_cutoff: Optional[float] = None) -> None:
        """
        Initialize the neighbor list.

        :param r_cutoff: Cutoff radius, defaults to None.
        :type r_cutoff: float, optional
        """
        self.r_cutoff = r_cutoff
        self.r_cutoff_updated: bool = False
        self.mask: Optional[jnp.ndarray] = None
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

    def update(self, structure) -> None:
        """
        This method updates the neighbor atom tensors including the number of neighbor and neighbor atom
        indices for the input structure.
        """
        # TODO: optimize updating the neighbor list, for example using the cell mesh, bin atoms (miniMD), etc.
        if self.r_cutoff is None:
            logger.debug("Skipped updating the neighbor list (no cutoff radius)")
            return

        if (not structure.requires_neighbor_update) and (not self.r_cutoff_updated):
            logger.debug("Skipped updating the neighbor list")
            return

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
