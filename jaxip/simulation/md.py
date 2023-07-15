from dataclasses import replace
from typing import Optional, Protocol, Tuple

import jax
import jax.numpy as jnp

from jaxip.atoms import Structure
from jaxip.atoms._structure import _calculate_center_of_mass
from jaxip.logger import logger
from jaxip.simulation.thermostat import BrendsenThermostat
from jaxip.types import Array, Element
from jaxip.units import units

KB: float = units.BOLTZMANN_CONSTANT


class PotentialInterface(Protocol):
    def __call__(self, structure: Structure) -> Array:
        ...

    def compute_forces(self, structure: Structure) -> Array:
        ...


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
    velocities: Array,
    masses: Array,
    positions: Array,
    forces: Array,
) -> Array:
    return (
        2 * _get_kinetic_energy(velocities, masses)
        + (positions * forces).sum()
    )


@jax.jit
def _get_verlet_new_positions(
    positions: Array,
    velocities: Array,
    forces: Array,
    time_step: Array,
) -> Array:
    return (
        positions
        + velocities * time_step
        + 0.5 * forces * time_step * time_step
    )


@jax.jit
def _get_verlet_new_velocities(
    velocities: Array,
    forces: Array,
    new_forces: Array,
    time_step: Array,
) -> Array:
    return velocities + 0.5 * (forces + new_forces) * time_step


def _get_potential_energy(
    potential: PotentialInterface,
    structure: Structure,
) -> Array:
    return potential(structure)


def _compute_forces(
    potential: PotentialInterface,
    structure: Structure,
) -> Array:
    return potential.compute_forces(structure)


class MDSimulator:
    def __init__(
        self,
        potential: PotentialInterface,
        initial_structure: Structure,
        time_step: float,
        initial_velocities: Optional[Array] = None,
        temperature: Optional[float] = None,
        thermostat: Optional[BrendsenThermostat] = None,
        atomic_masses: Optional[Array] = None,
        random_seed: int = 12345,
    ) -> None:
        """
        A molecular dynamics simulator to predict trajectory of particles over time.

        :param potential: input potential function
        :type potential: PotentialInterface
        :param initial_structure: initial structure of atoms
        :type initial_structure: Structure
        :param time_step: time step in Hartree time unit
        :type time_step: float
        :param initial_velocities: Initial atom velocities,
            otherwise it can be randomly generated from the input temperature, defaults to None
        :type initial_velocities: Optional[Array], optional
        :param temperature: input temperature which can be used for generating random velocities
            or setting target temperature for internally initialized thermostat if needed, defaults to None
        :type temperature: Optional[float], optional
        :param thermostat: input thermostat that controls the temperature of the system
            to the desired value, defaults to None
        :type thermostat: Optional[BrendsenThermostat], optional
        :param atomic_masses: atomic mass of atoms in the input structure, defaults to None
        :type atomic_masses: Optional[Array], optional
        :param random_seed: seed for generating random velocities, defaults to 12345
        :type random_seed: int, optional

        This simulator is intended to be used for small systems of atoms during
        the potential training in order to generate new dataset.
        """

        logger.debug(f"Initializing {self.__class__.__name__}")
        self.potential = potential
        self.time_step = time_step
        self.thermostat = thermostat

        self.masses: Array
        if atomic_masses is not None:
            self.masses = atomic_masses.reshape(-1, 1)
        else:
            logger.info("Extracting atomc masses from input structure")
            self.masses = initial_structure.get_masses().reshape(-1, 1)

        self.velocities: Array
        if initial_velocities is not None:
            self.velocities = initial_velocities
        else:
            assert (
                temperature is not None
            ), "At least one of input initial temperature or initial velocities must be given"
            logger.info(f"Generating random velocities ({temperature:0.2f} K)")
            self.velocities = self.generate_random_velocities(
                temperature, masses=self.masses, seed=random_seed
            )

        if (self.thermostat is None) and (temperature is not None):
            logger.info(f"Initializing thermostat ({temperature:0.2f} K)")
            logger.info(
                "Setting default thermostat constant to 100x time step"
            )
            self.thermostat = BrendsenThermostat(
                target_temperature=temperature,
                time_constant=100 * self.time_step,
            )

        self.step: int = 0
        self.elapsed_time: float = 0.0
        self.forces = _compute_forces(self.potential, initial_structure)
        self.temperature = _get_temperature(self.velocities, self.masses)
        self._structure = replace(initial_structure)

    def update(self) -> None:
        """Update parameters for next time step."""
        self.verlet_integration()
        self.step += 1
        self.elapsed_time += self.time_step
        self.temperature = _get_temperature(self.velocities, self.masses)
        if self.thermostat is not None:
            self.velocities = self.thermostat.get_rescaled_velocities(self)
            self.temperature = _get_temperature(self.velocities, self.masses)

    def verlet_integration(self) -> None:
        """Update atom positions and velocities based on Verlet algorithm."""
        arraylike_time_step = jnp.array(self.time_step)
        new_positions = _get_verlet_new_positions(
            self.positions, self.velocities, self.forces, arraylike_time_step
        )
        new_forces = _compute_forces(self.potential, self._structure)
        self.velocities = _get_verlet_new_velocities(
            self.velocities, self.forces, new_forces, arraylike_time_step
        )
        self.forces = new_forces
        self._structure = replace(self._structure, positions=new_positions)

    @classmethod
    def generate_random_velocities(
        cls,
        temperature: float,
        masses: Array,
        seed: int,
    ) -> Array:
        """Generate Maxwell-Boltzmann distributed random velocities."""
        key = jax.random.PRNGKey(seed)
        masses = masses.reshape(-1, 1)
        natoms = masses.shape[0]
        velocities = jax.random.normal(key, shape=(natoms, 3))
        velocities *= jnp.sqrt(
            temperature / _get_temperature(velocities, masses)
        )
        velocities -= _calculate_center_of_mass(velocities, masses)
        return velocities

    def repr_physical_params(self) -> str:
        """Represent current physical parameters."""
        return (
            f"{self.step:<10} "
            f"time[ps]:{units.TO_PICO_SECOND * self.elapsed_time:<10.5f} "
            f"Temp[K]:{self.temperature:<10.5f} "
            f"Etot[Ha]:{self.get_total_energy():<15.10f} "
            f"Epot[Ha]:{self.get_potential_energy():<15.10f} "
            f"Pres[kb]:{self.get_pressure() * units.TO_KILO_BAR:<10.5f}"
            if self._structure.box
            else ""
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
            forces=self.forces,
            total_energy=self.potential(self._structure),
        )

    def get_pressure(self) -> Array:
        assert (
            self._structure.box
        ), "Calulating pressure... input structure must have PBC box"
        volume = self._structure.box.volume
        virial = _get_virial(
            self.velocities,
            self.masses,
            self.positions,
            self.forces,
        )
        return virial / (3.0 * volume)  # type: ignore

    def get_center_of_mass_velocity(self) -> Array:
        return _calculate_center_of_mass(self.velocities, self.masses)

    def get_center_of_mass_position(self) -> Array:
        return _calculate_center_of_mass(self.positions, self.masses)

    def get_potential_energy(self) -> Array:
        return _get_potential_energy(self.potential, self._structure)

    def get_kinetic_energy(self) -> Array:
        return _get_kinetic_energy(self.velocities, self.masses)

    def get_total_energy(self) -> Array:
        return self.get_potential_energy() + self.get_kinetic_energy()
