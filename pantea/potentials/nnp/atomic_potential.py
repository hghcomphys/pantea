from dataclasses import dataclass

from frozendict import frozendict

from pantea.atoms.structure import Structure
from pantea.descriptors.acsf.acsf import ACSF
from pantea.descriptors.scaler import DescriptorScaler
from pantea.models.nn.network import NeuralNetworkModel
from pantea.potentials.nnp.energy import _compute_energy_per_atom
from pantea.types import Array, Element


@dataclass(frozen=True)
class AtomicPotential:
    """
    Atomic potential.

    This class simply chains all the required transformers (descriptor, scaler, model, etc.)
    to calculate per-atom energy for a specific element.
    """

    descriptor: ACSF
    scaler: DescriptorScaler
    model: NeuralNetworkModel

    def apply(
        self,
        params: frozendict[str, Array],
        structure: Structure,
    ) -> Array:
        """
        Calculate model output energy.
        It must be noted model output for each element has no physical meaning.
        It's only the total energy, sum of the atomic energies over all atom in the structure,
        which has actually physical meaning.

        :param params: model parameters
        :param structure: input structure
        :return: model energy output
        """
        element: Element = self.descriptor.central_element  # type: ignore
        aid: Array = structure.select(element)

        return _compute_energy_per_atom(
            self,
            structure.positions[aid],
            params,
            structure.get_inputs_per_element()[element],
        )

    @property
    def model_input_size(self) -> int:
        """Return size of the model input."""
        return self.descriptor.num_symmetry_functions

    def __repr__(self) -> str:
        out: str = f"{self.__class__.__name__}("
        for component in self.__annotations__:
            out += f"\n  {component}={getattr(self, component)},"
        out += "\n)"
        return out
