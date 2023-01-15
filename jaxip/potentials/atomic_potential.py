from dataclasses import dataclass

from frozendict import frozendict

from jaxip.descriptors.base import Descriptor
from jaxip.descriptors.scaler import DescriptorScaler
from jaxip.models.nn import NeuralNetworkModel
from jaxip.potentials._atomic_energy import _compute_atomic_energy
from jaxip.structure.structure import Structure
from jaxip.types import Array


@dataclass(frozen=True)
class AtomicPotential:
    """Atomic potential."""

    descriptor: Descriptor
    scaler: DescriptorScaler
    model: NeuralNetworkModel

    def apply(self, params: frozendict, structure: Structure) -> Array:
        """
        Calculate model output energy.
        It must be noted this has no physical meaning but only the total energy.

        :param params: model parameters
        :type params: frozendict
        :param structure: input structure
        :type structure: Structure
        :return: model energy output
        :rtype: Array
        """
        element = self.descriptor.element  # type: ignore FIXME:
        aid = structure.select(element)

        return _compute_atomic_energy(
            self,
            structure.position[aid],
            params,
            structure.get_inputs()[element],
        )
