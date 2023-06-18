from dataclasses import replace
from typing import Optional, Protocol, Tuple

import jax
import jax.numpy as jnp

from jaxip.atoms import Structure
from jaxip.logger import logger
from jaxip.simulation.thermostat import BrendsenThermostat
from jaxip.types import Array, Element
from jaxip.units import units

KB = units.BOLTZMANN_CONSTANT


class PotentialInterface(Protocol):
    def __call__(self, structure: Structure) -> Array:
        ...

    def compute_force(self, structure: Structure) -> Array:
        ...


@jax.jit
def _get_center_of_mass(array: Array, mass: Array) -> Array:
    return jnp.sum(mass * array, axis=0) / jnp.sum(mass)


@jax.jit
def _get_kinetic_energy(velocity: Array, mass: Array) -> Array:
    return 0.5 * jnp.sum(mass * velocity * velocity)


@jax.jit
def _get_temperature(velocity: Array, mass: Array) -> Array:
    kinetic_energy = _get_kinetic_energy(velocity, mass)
    natoms = velocity.shape[0]
    return 2 * kinetic_energy / (3 * natoms * KB)


@jax.jit
def _get_virial_term(
    velocity: Array,
    mass: Array,
    position: Array,
    force: Array,
) -> Array:
    return 2 * _get_kinetic_energy(velocity, mass) + (position * force).sum()


def _get_potential_energy(
    potential: PotentialInterface,
    structure: Structure,
) -> Array:
    return potential(structure)


def _compute_force(
    potential: PotentialInterface,
    structure: Structure,
) -> Array:
    return potential.compute_force(structure)


class MDSimulator:
    def __init__(
        self,
        potential: PotentialInterface,
        initial_structure: Structure,
        time_step: float,
        initial_velocity: Optional[Array] = None,
        temperature: Optional[float] = None,
        thermostat: Optional[BrendsenThermostat] = None,
        atomic_mass: Optional[Array] = None,
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
        :param initial_velocity: Initial velocity of atom,
            otherwise it can be randomly generated from the input temperature, defaults to None
        :type initial_velocity: Optional[Array], optional
        :param temperature: input temperature which can be used for generating random velocity
            or setting target temperature for internally initialized thermostat if needed, defaults to None
        :type temperature: Optional[float], optional
        :param thermostat: input thermostat that controls the temperature of the system
            to the desired value, defaults to None
        :type thermostat: Optional[BrendsenThermostat], optional
        :param atomic_mass: atomic mass of atoms in the input structure, defaults to None
        :type atomic_mass: Optional[Array], optional
        :param random_seed: seed for generating random velocities, defaults to 12345
        :type random_seed: int, optional

        This simulator is intended to be used for small systems of atoms during
        the potential training in order to generate new dataset.
        """

        logger.debug(f"Initializing {self.__class__.__name__}")
        self.potential = potential
        self.time_step = time_step
        self.thermostat = thermostat

        self.mass: Array
        if atomic_mass is not None:
            self.mass = atomic_mass.reshape(-1, 1)
        else:
            logger.info("Extracting atomc masses from input structure")
            self.mass = initial_structure.mass.reshape(-1, 1)

        self.velocity: Array
        if initial_velocity is not None:
            self.velocity = initial_velocity
        else:
            assert (
                temperature is not None
            ), "At least one of initial temperature or initial velocity must be given"
            logger.info(f"Generating random velocities ({temperature:0.2f} K)")
            self.velocity = self.generate_random_velocity(
                temperature, mass=self.mass, seed=random_seed
            )

        if (self.thermostat is None) and (temperature is not None):
            logger.info(f"Initializing thermostat ({temperature:0.2f} K)")
            logger.info("Setting default thermostat constant to 100x timestep")
            self.thermostat = BrendsenThermostat(
                target_temperature=temperature,
                time_constant=100 * self.time_step,
            )

        self.step: int = 0
        self.elapsed_time: float = 0.0
        self.force = _compute_force(self.potential, initial_structure)
        self.temperature = _get_temperature(self.velocity, self.mass)
        self._structure = replace(initial_structure)

    def run_simulation(
        self,
        num_steps: int = 1,
        output_freq: Optional[int] = None,
    ) -> None:
        """Run molecular simulation for a given number of steps."""
        if output_freq is None:
            output_freq = 1 if num_steps < 100 else int(0.01 * num_steps)
        is_output = output_freq > 0
        init_step = self.step
        try:
            for _ in range(num_steps):
                if is_output and ((self.step - init_step) % output_freq == 0):
                    print(self.repr_physical_params())
                self.molecular_dynamics_step()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
        if is_output:
            print(self.repr_physical_params())

    def molecular_dynamics_step(self) -> None:
        """Update parameters for next time step."""
        self.verlet_integration()
        self.step += 1
        self.elapsed_time += self.time_step
        self.temperature = _get_temperature(self.velocity, self.mass)
        if self.thermostat is not None:
            self.velocity = self.thermostat.get_rescaled_velocity(self)

    def verlet_integration(self) -> None:
        """Update atom positions and velocities based on Verlet algorithm."""
        new_position = (
            self.position
            + self.velocity * self.time_step
            + 0.5 * self.force * self.time_step * self.time_step
        )
        new_force = _compute_force(self.potential, self._structure)
        self.velocity += 0.5 * (self.force + new_force) * self.time_step
        self.force = new_force
        self._structure = replace(self._structure, position=new_position)

    @classmethod
    def generate_random_velocity(
        cls,
        temperature: float,
        mass: Array,
        seed: int,
    ) -> Array:
        """Generate Maxwell-Boltzmann distributed random velocities."""
        key = jax.random.PRNGKey(seed)
        mass = mass.reshape(-1, 1)
        natoms = mass.shape[0]
        velocity = jax.random.normal(key, shape=(natoms, 3))
        velocity *= jnp.sqrt(temperature / _get_temperature(velocity, mass))
        velocity -= _get_center_of_mass(velocity, mass)
        return velocity

    def repr_physical_params(self) -> str:
        return (
            f"{self.step:<10} "
            f"time[ps]:{units.TO_PICO_SECOND * self.elapsed_time:<10.5f} "
            f"Temp[K]:{self.temperature:<15.10f} "
            f"Etot[Ha]:{self.get_total_energy():<15.10f} "
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
            force=self.force,
            total_energy=self.potential(self._structure),
        )

    def get_pressure(self) -> Array:
        assert (
            self._structure.box
        ), "Calulating pressure... input structure must have PBC box"
        volume = self._structure.box.volume
        virial = _get_virial_term(self.velocity, self.mass, self.position, self.force)
        return virial / (3.0 * volume)  # type: ignore

    def get_com_velocity(self) -> Array:
        """Return center of mass velocity."""
        return _get_center_of_mass(self.velocity, self.mass)

    def get_com_position(self) -> Array:
        """Return center of mass position."""
        return _get_center_of_mass(self.position, self.mass)

    def get_potential_energy(self) -> Array:
        return _get_potential_energy(self.potential, self._structure)

    def get_kinetic_energy(self) -> Array:
        return _get_kinetic_energy(self.velocity, self.mass)

    def get_total_energy(self) -> Array:
        return self.get_potential_energy() + self.get_kinetic_energy()
