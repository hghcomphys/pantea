from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Protocol, Tuple

import jax
import jax.numpy as jnp

from pantea.atoms import Structure
from pantea.atoms.element import ElementMap
from pantea.logger import logger
from pantea.simulation.thermostat import BrendsenThermostat
from pantea.types import Array, Element
from pantea.units import units

KB: float = units.BOLTZMANN_CONSTANT


@jax.jit
def _get_kinetic_energy(velocities: Array, masses: Array) -> Array:
    return 0.5 * jnp.sum(masses * velocities * velocities)


@jax.jit
def _get_temperature(velocities: Array, masses: Array) -> Array:
    kinetic_energy = _get_kinetic_energy(velocities, masses)
    natoms = velocities.shape[0]
    return 2 * kinetic_energy / (3 * natoms * KB)


@jax.jit
def _get_virial(
    velocities: Array, masses: Array, positions: Array, forces: Array
) -> Array:
    return 2 * _get_kinetic_energy(velocities, masses) + jnp.sum(positions * forces)


@jax.jit
def _get_verlet_new_positions(
    positions: Array, velocities: Array, forces: Array, time_step: Array
) -> Array:
    return positions + velocities * time_step + 0.5 * forces * time_step * time_step


@jax.jit
def _get_verlet_new_velocities(
    velocities: Array,
    forces: Array,
    new_forces: Array,
    time_step: Array,
) -> Array:
    return velocities + 0.5 * (forces + new_forces) * time_step


@jax.jit
def _calculate_center_of_mass(array: Array, masses: Array) -> Array:
    return jnp.sum(masses * array, axis=0) / jnp.sum(masses)


class PotentialInterface(Protocol):
    def __call__(self, structure: Structure) -> Array:
        ...

    def compute_forces(self, structure: Structure) -> Array:
        ...


def _get_potential_energy(potential: PotentialInterface, structure: Structure) -> Array:
    return potential(structure)


def _compute_forces(potential: PotentialInterface, structure: Structure) -> Array:
    return potential.compute_forces(structure)


@dataclass
class MDSystem:
    structure: Structure
    velocities: Array
    masses: Array
    potential: PotentialInterface

    def __post_init__(self) -> None:
        forces = _compute_forces(self.potential, self.structure)
        self.structure = replace(self.structure, forces=forces)

    @classmethod
    def from_structure(
        cls,
        structure: Structure,
        temperature: float,
        potential: PotentialInterface,
        seed: int = 2023,
    ) -> MDSystem:
        logger.debug(f"Creating {cls.__name__} from Structure")
        masses = ElementMap.get_masses_from_structure(structure).reshape(-1, 1)
        velocities = cls.generate_random_velocities(temperature, masses, seed)
        return cls(structure, velocities, masses, potential)

    @classmethod
    def generate_random_velocities(
        cls,
        temperature: float,
        masses: Array,
        seed: int,
    ) -> Array:
        """Generate Maxwell-Boltzmann distributed random velocities."""
        key = jax.random.PRNGKey(seed)
        natoms = masses.shape[0]
        velocities = jax.random.normal(key, shape=(natoms, 3))
        velocities *= jnp.sqrt(temperature / _get_temperature(velocities, masses))
        velocities -= _calculate_center_of_mass(velocities, masses)
        return velocities

    def get_elements(self) -> Tuple[Element, ...]:
        return self.structure.get_elements()

    def get_structure(self) -> Structure:
        return replace(self.structure, total_energy=self.potential(self.structure))

    def get_pressure(self) -> Array:
        box = self.structure.box
        assert (
            box is not None
        ), "Calculating pressure... input structure must have PBC box"
        virial = _get_virial(self.velocities, self.masses, self.positions, self.forces)
        return virial / (3.0 * box.volume)  # type: ignore

    def get_temperature(self) -> Array:
        return _get_temperature(self.velocities, self.masses)

    def get_center_of_mass_velocity(self) -> Array:
        return _calculate_center_of_mass(self.velocities, self.masses)

    def get_center_of_mass_position(self) -> Array:
        return _calculate_center_of_mass(self.positions, self.masses)

    def get_potential_energy(self) -> Array:
        return _get_potential_energy(self.potential, self.structure)

    def get_kinetic_energy(self) -> Array:
        return _get_kinetic_energy(self.velocities, self.masses)

    def get_total_energy(self) -> Array:
        return self.get_potential_energy() + self.get_kinetic_energy()

    @property
    def positions(self) -> Array:
        return self.structure.positions

    @property
    def forces(self) -> Array:
        return self.structure.forces

    @property
    def natoms(self) -> int:
        return self.structure.natoms


class MDSimulator:
    def __init__(
        self,
        time_step: float,
        thermostat: Optional[BrendsenThermostat] = None,
    ) -> None:
        """
        A molecular dynamics simulator to predict trajectory of particles over time.
        :param time_step: time step in Hartree time unit
        :type time_step: float
        :param thermostat: input thermostat that controls the temperature of the system
            to the desired value, defaults to None
        :type thermostat: Optional[BrendsenThermostat], optional

        This simulator is intended to be used for small systems of atoms during
        the potential training in order to generate new training dataset.
        """

        self.time_step = time_step
        self.thermostat = thermostat
        self.step: int = 0
        self.elapsed_time: float = 0.0

    def simulate_one_step(self, system: MDSystem) -> None:
        """Update parameters for next time step."""
        self.verlet_integration(system)
        self.step += 1
        self.elapsed_time += self.time_step
        self.temperature = _get_temperature(system.velocities, system.masses)
        if self.thermostat is not None:
            system.velocities = self.thermostat.get_rescaled_velocities(self, system)
            self.temperature = _get_temperature(system.velocities, system.masses)

    def verlet_integration(self, system: MDSystem) -> None:
        """Update atom positions and velocities based on Verlet algorithm."""
        time_step = jnp.array(self.time_step)
        new_positions = _get_verlet_new_positions(
            system.positions, system.velocities, system.forces, time_step
        )
        system.structure = replace(system.structure, positions=new_positions)
        new_forces = _compute_forces(system.potential, system.structure)
        system.velocities = _get_verlet_new_velocities(
            system.velocities, system.forces, new_forces, time_step
        )
        system.structure.forces = new_forces

    def repr_physical_params(self, system: MDSystem) -> str:
        """Represent current physical parameters."""
        return (
            f"{self.step:<10} "
            f"time[ps]:{units.TO_PICO_SECOND * self.elapsed_time:<10.5f} "
            f"Temp[K]:{system.get_temperature():<10.5f} "
            f"Etot[Ha]:{system.get_total_energy():<15.10f} "
            f"Epot[Ha]:{system.get_potential_energy():<15.10f} "
            f"Pres[kb]:{system.get_pressure() * units.TO_KILO_BAR:<10.5f}"
            if system.structure.box
            else ""
        )
