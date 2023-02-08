from dataclasses import dataclass, field
from typing import Optional

import jax.numpy as jnp

from jaxip.base import _BaseJaxPytreeDataClass, register_jax_pytree_node
from jaxip.logger import logger
from jaxip.structure._neighbor import _calculate_cutoff_mask
from jaxip.types import Array


@dataclass
class Neighbor(_BaseJaxPytreeDataClass):
    """
    Neighbor creates a neighbor list of atoms for an input structure
    and it is by design independent of `Structure`.

    .. note::
        For MD simulations, re-neighboring the list is required every few steps.
        This is usually implemented together with defining a skin radius.
    """

    # Array type attributes must be define first (with type hint)
    mask: Optional[Array] = None
    r_cutoff: Optional[float] = None
    r_cutoff_updated: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """Post initialize the neighbor list."""
        logger.debug(f"Initializing {self}")

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    def set_cutoff_radius(self, r_cutoff: float) -> None:
        """
        Set a given cutoff radius for the neighbor list.
        The neighbor list will be updated on the first call.

        :param r_cutoff: A new cutoff radius
        :type r_cutoff: float
        """
        logger.debug(
            f"Setting Neighbor cutoff radius from {self.r_cutoff} to {r_cutoff}"
        )
        self.r_cutoff = r_cutoff
        self.r_cutoff_updated = True

    def update(self, structure) -> None:
        """
        Update the list neighboring atoms.

        It is based on mask approach which is different from the conventional methods
        for updating the neighbor list (e.g. by defining neighbor indices).
        It's many due to the fact that jax execute quite fast on vectorized presentations
        rather than simple looping in python (jax.jit).

        .. note::
            Further adjustments can be added regarding the neighbor list updating methods.
            But for time being the mask-based approach works well on `JAX`.
        """
        # TODO: optimize updating the neighbor list, for example using the cell mesh, bin atoms (miniMD), etc.
        if self.r_cutoff is None:
            logger.debug("Skipped updating the neighbor list (no cutoff radius)")
            return

        if (not structure.requires_neighbor_update) and (not self.r_cutoff_updated):
            logger.debug("Skipped updating the neighbor list")
            return

        logger.debug("Updating the neighbor list")
        self.mask = _calculate_cutoff_mask(
            structure,
            jnp.atleast_1d(self.r_cutoff),
        )

        # Avoid updating the neighbor list for the next call
        structure.requires_neighbor_update = False
        self.r_cutoff_updated = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r_cutoff={self.r_cutoff})"


register_jax_pytree_node(Neighbor)
