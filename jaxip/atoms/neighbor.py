from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp

from jaxip.atoms._neighbor import _calculate_cutoff_masks
from jaxip.logger import logger
from jaxip.pytree import BaseJaxPytreeDataClass, register_jax_pytree_node
from jaxip.types import Array


@dataclass
class Neighbor(BaseJaxPytreeDataClass):
    """
    Create a neighbor list of atoms for structure.
    and it is by design independent of `Structure`.

    .. note::
        For MD simulations, re-neighboring the list is required every few steps.
        This is usually implemented together with defining a skin radius.
    """

    masks: Optional[Array] = None
    r_cutoff: Optional[float] = None

    def __post_init__(self) -> None:
        """Post initialize the neighbor list."""
        logger.debug(f"Initializing {self}")
        self._assert_jit_dynamic_attributes(expected=("masks",))
        self._assert_jit_static_attributes(expected=("r_cutoff",))

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
            f"Setting Neighbor cutoff radius from "
            f"{self.r_cutoff} to {r_cutoff}"
        )
        self.r_cutoff = r_cutoff

    def update(self, structure) -> None:
        """
        Update the list of neighboring atoms.

        It is based on mask approach which is different from the conventional methods
        for updating the neighbor list (e.g. by defining neighbor indices).
        It's due to the fact that jax execute quite fast on vectorized variables
        rather than simple looping in python (jax.jit).

        .. note::
            Further adjustments can be added regarding the neighbor list updating methods.
            But for time being the mask-based approach works well on `JAX`.
        """
        if self.r_cutoff is None:
            logger.debug(
                "Skipped updating the neighbor list"
                f"(r_cutoff={self.r_cutoff})"
            )
            return

        logger.debug("Updating neighbor list")
        self.mask = _calculate_cutoff_masks(
            structure,
            jnp.atleast_1d(self.r_cutoff),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r_cutoff={self.r_cutoff})"


register_jax_pytree_node(Neighbor)
