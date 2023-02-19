from dataclasses import dataclass

from frozendict import frozendict

from jaxip.descriptors.base import Descriptor
from jaxip.descriptors.scaler import DescriptorScaler
from jaxip.models.nn import NeuralNetworkModel
from jaxip.potentials._energy import _compute_atomic_energy
from jaxip.structure.structure import Structure
from jaxip.types import Array


@dataclass(frozen=True)
class AtomicPotential:
    """
    Atomic potential.

    It chains all the required transformers (descriptor, scaler, model, etc.)
    to calculate per-atom energy.
    """

    descriptor: Descriptor
    scaler: DescriptorScaler
    model: NeuralNetworkModel

    def apply(self, params: frozendict, structure: Structure) -> Array:
        """
        Calculate model output energy.
        It must be noted model output for each element has no physical meaning.
        It's only the total energy, sum of the atomic energies over all atom in the structure,
        which has physical meaning.

        :param params: model parameters
        :type params: frozendict
        :param structure: input structure
        :type structure: Structure
        :return: model energy output
        :rtype: Array
        """
        element = self.descriptor.element  # type: ignore
        aid = structure.select(element)

        return _compute_atomic_energy(
            self,
            structure.position[aid],
            params,
            structure.get_inputs()[element],
        )

    @property
    def model_input_size(self) -> int:
        """Return size of the model input."""
        return self.descriptor.num_descriptors

    def __repr__(self) -> str:
        out: str = f"{self.__class__.__name__}("
        for component in self.__annotations__:
            out += f"\n  {component}={getattr(self, component)},"
        out += "\n)"
        return out
