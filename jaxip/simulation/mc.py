from dataclasses import replace
from typing import Optional, Protocol, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxip.atoms import Structure
from jaxip.logger import logger
from jaxip.types import Array, Element
from jaxip.units import units

KB = units.BOLTZMANN_CONSTANT


class PotentialInterface(Protocol):
    def __call__(self, structure: Structure) -> Array:
        ...

    def compute_force(self, structure: Structure) -> Array:
        ...


def _compute_potential_energy(
    potential: PotentialInterface,
    structure: Structure,
) -> Array:
    return potential(structure)


class MCSimulator:
    def __init__(
        self,
        potential: PotentialInterface,
        initial_structure: Structure,
        temperature: float,
        translate_step: float,
        atomic_mass: Optional[Array] = None,
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
        :param atomic_mass: atomic mass of atoms in the input structure, defaults to None
        :type atomic_mass: Optional[Array], optional
        :param random_seed: seed for generating random velocities, defaults to 12345
        :type random_seed: int, optional

        This simulator is intended to be used for small systems of atoms during
        the potential training in order to generate new dataset.
        """

        logger.debug(f"Initializing {self.__class__.__name__}")
        self.potential = potential
        self.temperature = temperature
        self.translate_step = translate_step

        self.mass: Array
        if atomic_mass is not None:
            self.mass = atomic_mass.reshape(-1, 1)
        else:
            logger.info("Extracting atomc masses from input structure")
            self.mass = initial_structure.mass.reshape(-1, 1)

        self.step: int = 0
        self.energy = _compute_potential_energy(self.potential, initial_structure)
        self._structure = replace(initial_structure)
        np.random.seed(random_seed)

    def run_simulation(
        self,
        num_steps: int = 1,
        output_freq: Optional[int] = None,
    ) -> None:
        """Run Monte carlo simulation for a given number of steps."""
        if output_freq is None:
            output_freq = 1 if num_steps < 100 else int(0.01 * num_steps)
        is_output = output_freq > 0
        init_step = self.step
        try:
            for _ in range(num_steps):
                if is_output and ((self.step - init_step) % output_freq == 0):
                    print(self.repr_physical_params())
                self.monte_carlo_step()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
        if is_output:
            print(self.repr_physical_params())

    def monte_carlo_step(self) -> None:
        """Update parameters for next time step."""
        self.metropolis_algorithm()
        self.step += 1

    def metropolis_algorithm(self) -> None:
        """Update atom positions based on metropolis algorithm."""
        displacement = np.random.uniform(
            low=-self.translate_step,
            high=self.translate_step,
            size=(3,),
        )
        atom_index = np.random.randint(low=0, high=self.natoms)
        new_position = self._structure.position.at[atom_index].add(displacement)
        new_structure = replace(self._structure, position=new_position)
        new_energy = _compute_potential_energy(self.potential, new_structure)

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
        return (
            f"{self.step:<10} "
            f"Temp[K]:{self.temperature:<15.10f} "
            f"Epot[Ha]:{self.get_potential_energy():<15.10f} "
        )

    @property
    def position(self) -> Array:
        return self._structure.position

    @property
    def natoms(self) -> int:
        return self._structure.natoms

    @property
    def elements(self) -> Tuple[Element]:
        return self._structure.elements

    def get_structure(self) -> Structure:
        return replace(
            self._structure,
            total_energy=self.potential(self._structure),
        )

    def get_potential_energy(self) -> Array:
        return _compute_potential_energy(self.potential, self._structure)
