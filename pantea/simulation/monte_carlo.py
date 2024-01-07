from dataclasses import replace
from typing import Protocol

import jax.numpy as jnp
import numpy as np

from pantea.atoms.structure import Structure
from pantea.logger import logger
from pantea.simulation.system import System
from pantea.types import Array
from pantea.units import units

KB: float = units.BOLTZMANN_CONSTANT


class PotentialInterface(Protocol):
    def __call__(self, structure: Structure) -> Array:
        ...

    def compute_forces(self, structure: Structure) -> Array:
        ...


class MCSimulator:
    def __init__(
        self,
        translate_step: float,
        target_temperature: float,
        movements_per_step: int = 1,
        seed: int = 12345,
    ) -> None:
        """
        A Monte Carlo simulator to predict atom configurations for a given temperature.

        :param translate_step: maximum translate step in Bohr unit
        :type translate_step: float
        :param target_temperature: target temperature for the system
        :type target_temperature: float
        :param atomic_masses: atomic mass of atoms in the input structure, defaults to None
        :type atomic_masses: Optional[Array], optional
        :param movements_per_step: number of random movements per step
        :type movements_per_step: int, default to 1
        :param seed: seed for generating random atom movements, defaults to 12345
        :type seed: int, optional

        It must be noted that this MC simulator modifies atom
        `positions` and `total_energy` of Structure inside the system.

        This simulator is intended to be used for small systems of atoms during
        the potential training in order to generate new dataset.
        """

        logger.debug(f"Initializing {self.__class__.__name__}")
        self.translate_step = translate_step
        self.target_temperature = target_temperature
        self.movements_per_step = movements_per_step
        self.step: int = 0
        np.random.seed(seed)

    def simulate_one_step(self, system: System) -> None:
        """Update parameters for next time step."""
        self.metropolis_algorithm(system)
        self.step += 1

    def metropolis_algorithm(self, system: System) -> None:
        """Update atom positions based on metropolis algorithm."""
        displacements = np.random.uniform(
            low=-self.translate_step,
            high=self.translate_step,
            size=(self.movements_per_step, 3),
        )
        atom_indices = np.random.randint(
            low=0, high=system.natoms, size=(self.movements_per_step,)
        )
        new_positions = system.positions.at[atom_indices].add(displacements)
        new_structure = replace(
            system.structure, positions=new_positions
        )  # shallow copy
        new_energy = system.potential(new_structure)

        energy = system.structure.total_energy
        accept: bool = False
        if new_energy <= energy:
            accept = True
        else:
            prob = jnp.exp(-(new_energy - energy) / (KB * self.target_temperature))
            accept = True if prob >= np.random.uniform(0.0, 1.0) else False

        if accept:
            system.structure.total_energy = new_energy
            system.structure.positions = new_structure.positions

    def repr_physical_params(self, system: System) -> str:
        """Represent current physical parameters."""
        return f"{self.step:<10} Epot[Ha]:{system.get_potential_energy():<15.10f} "
