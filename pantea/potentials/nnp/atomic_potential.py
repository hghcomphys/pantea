from dataclasses import dataclass

from pantea.atoms.structure import Structure
from pantea.descriptors.acsf.acsf import ACSF
from pantea.descriptors.scaler import DescriptorScaler
from pantea.models.nn.network import NeuralNetworkModel
from pantea.potentials.nnp.energy import ModelParams, _compute_energy_per_atom
from pantea.types import Array


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
        model_params: ModelParams,
        structure: Structure,
    ) -> Array:
        """
        Calculate model output energy for a specific element.

        It must be noted that the model output has no physical meaning.
        It's only the total energy, sum of the atomic energies over all atom,
        which has actually physical meaning.

        :param model_params: model parameters per element
        :param structure: input structure
        :return: model energy output
        """
        atom_index = structure.select(self.descriptor.central_element)
        return _compute_energy_per_atom(
            self,
            structure.positions[atom_index],
            model_params,
            structure.as_kernel_args(),
        )  # type: ignore

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
