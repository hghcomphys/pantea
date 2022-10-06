from ..logger import logger
from ..base import BaseTorchipClass
from ..config import dtype
from ..utils.attribute import set_as_attribute

# from .structure import Structure  # TODO: circular import error
import torch


class Neighbor(BaseTorchipClass):
    """
    Neighbor class creates a neighbor list of atom for the input structure.
    Neighbor is teated as a buffer which classes are responsible to prepare it before using.

    It is designed to be independent of the input structure.
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
        self.tensors = None
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

    def _init_tensors(self, structure) -> None:
        """
        Allocating tensors for the neighbor list from the input structure.

        :param structure: An instance of Structure
        :type structure: Dict[Tensor]
        """
        # TODO: reduce natoms*natoms tensor size!
        # TODO: define max_num_neighbor to avoid extra memory allocation!

        # Avoid re-allocating structure with the same size
        try:
            if structure.natoms == len(self.number):
                logger.debug(f"Avoid re-allocating the neighbor tensors")
                return
        except AttributeError:
            pass

        # --------------------------
        logger.debug("Allocating tensors for neighbor list")
        self.tensors = {}

        # Neighbor atoms numbers and indices
        self.tensors["number"] = torch.empty(
            structure.natoms, dtype=dtype.UINT, device=structure.device
        )
        self.tensors["index"] = torch.empty(
            structure.natoms,
            structure.natoms,
            dtype=dtype.INDEX,
            device=structure.device,
        )

        # Logging
        for attr, tensor in self.tensors.items():
            logger.debug(
                f"{attr:12} -> Tensor(shape='{tensor.shape}', dtype='{tensor.dtype}', device='{tensor.device}')"
            )

        set_as_attribute(self, self.tensors)

    def update(self, structure) -> None:
        """
        This method updates the neighbor atom tensors including the number of neighbor and neighbor atom indices
        within the input structure.
        """
        # TODO: optimize updating the neighbor list, for example using the cell mesh, bin atoms (miniMD), etc.
        if not self.r_cutoff:
            logger.debug("Skipping updating the neighbor list (no cutoff radius)")
            return

        if not structure.requires_neighbor_update and not self.r_cutoff_updated:
            logger.debug("Skipping updating the neighbor list")
            return

        self._init_tensors(structure)

        # ----------------------------------------
        logger.debug("Updating neighbor list")

        # TODO: define staticmethod _update()
        # Tensors no need to be differentiable here
        with torch.no_grad():
            nn = self.number
            ni = self.index
            for aid in range(
                structure.natoms
            ):  # TODO: optimization: torch unbind or vmap
                rij = structure.calculate_distance(aid)
                # Get atom indices within the cutoff radius
                ni_ = torch.nonzero(rij < self.r_cutoff, as_tuple=True)[0]
                # Remove self-counting atom index
                ni_ = ni_[ni_ != aid]
                # Set neighbor list tensors
                nn[aid] = ni_.shape[0]
                ni[aid, : nn[aid]] = ni_

        # Avoid updating the neighbor list the next time
        structure.requires_neighbor_update = False
        self.r_cutoff_updated = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r_cutoff={self.r_cutoff})"
