from dataclasses import replace
from typing import Optional, Protocol, Tuple

import jax.numpy as jnp
import numpy as np

from jaxip.atoms.structure import Structure
from jaxip.logger import logger
from jaxip.types import Array, Element
from jaxip.units import units

KB: float = units.BOLTZMANN_CONSTANT


class PotentialInterface(Protocol):
    def __call__(self, structure: Structure) -> Array:
        ...

    def compute_forces(self, structure: Structure) -> Array:
        ...


class MCSimulator:
    def __init__(
        self,
        potential: PotentialInterface,
        initial_structure: Structure,
        temperature: float,
        translate_step: float,
        movements_per_step: int = 1,
        atomic_masses: Optional[Array] = None,
        random_seed: int = 12345,
    ) -> None:
        """
        A Monte Carlo simulator to predict configuration space of particles.

        :param potential: input potential function
        :type potential: PotentialInterface
        :param initial_structure: initial structure of atoms
        :type initial_structure: Structure
        :param translate_step: maximum translate step in Bohr unit
        :type time_step: float
        :param temperature: desired temperature of the system
        :type temperature: float
        :param atomic_masses: atomic mass of atoms in the input structure, defaults to None
        :type atomic_masses: Optional[Array], optional
        :param random_seed: seed for generating random atom movements, defaults to 12345
        :type random_seed: int, optional

        This simulator is intended to be used for small systems of atoms during
        the potential training in order to generate new dataset.
        """

        logger.debug(f"Initializing {self.__class__.__name__}")
        self.potential = potential
        self.temperature = temperature
        self.translate_step = translate_step
        self.movements_per_step = movements_per_step

        self.masses: Array
        if atomic_masses is not None:
            self.masses = atomic_masses.reshape(-1, 1)
        else:
            logger.info("Extracting atomc masses from input structure")
            self.masses = initial_structure.get_masses().reshape(-1, 1)

        self.step: int = 0
        self.energy = self.potential(initial_structure)
        self._structure = replace(initial_structure)
        np.random.seed(random_seed)

    def update(self) -> None:
        """Update parameters for next time step."""
        self.metropolis_algorithm()
        self.step += 1

    def metropolis_algorithm(self) -> None:
        """Update atom positions based on metropolis algorithm."""
        displacements = np.random.uniform(
            low=-self.translate_step,
            high=self.translate_step,
            size=(self.movements_per_step, 3),
        )
        atom_indices = np.random.randint(
            low=0, high=self.natoms, size=(self.movements_per_step,)
        )
        new_position = self._structure.positions.at[atom_indices].add(displacements)
        new_structure = replace(self._structure, positions=new_position)
        new_energy = self.potential(new_structure)

        accept: bool = False
        if new_energy <= self.energy:
            accept = True
        else:
            prob = jnp.exp(-(new_energy - self.energy) / (KB * self.temperature))
            accept = True if prob >= np.random.uniform(0.0, 1.0) else False

        if accept:
            self.energy = new_energy
            self._structure = new_structure

    def repr_physical_params(self) -> str:
        """Represent current physical parameters."""
        return (
            f"{self.step:<10} "
            f"Temp[K]:{self.temperature:<15.10f} "
            f"Epot[Ha]:{self.get_potential_energy():<15.10f} "
        )

    @property
    def positions(self) -> Array:
        return self._structure.positions

    @property
    def natoms(self) -> int:
        return self._structure.natoms

    def get_elements(self) -> Tuple[Element]:
        return self._structure.get_elements()

    def get_structure(self) -> Structure:
        return replace(
            self._structure,
            total_energy=self.potential(self._structure),
        )

    def get_potential_energy(self) -> Array:
        return self.potential(self._structure)
