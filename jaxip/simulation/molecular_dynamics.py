from dataclasses import replace
from typing import Optional, Protocol, Tuple, cast

import jax
import jax.numpy as jnp

from jaxip.atoms import Structure
from jaxip.logger import logger
from jaxip.types import Array, Element


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
    return 0.5 * jnp.sum(mass * velocity**2)


@jax.jit
def _get_temperature(velocity: Array, mass: Array) -> Array:
    KB = 3.166811563e-6  # Boltzmann constant in Hartree/K
    kinetic_energy = _get_kinetic_energy(velocity, mass)
    natoms = velocity.shape[0]
    return 2 * kinetic_energy / (3 * natoms * KB)


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
    """A basic molecular dynamics simulator for a given potential function."""

    def __init__(
        self,
        potential: PotentialInterface,
        initial_structure: Structure,
        time_step: float,
        target_temperature: Optional[float] = None,
        initial_velocity: Optional[Array] = None,
        thermostat_time_constant: Optional[float] = None,
        atomic_mass: Optional[Array] = None,
        random_seed: int = 12345,
    ):
        logger.debug(f"Initializing {self.__class__.__name__}")
        self.potential = potential
        self.structure = initial_structure
        self.time_step = time_step
        self.target_temperature = target_temperature
        self.force = _compute_force(self.potential, self.structure)

        self.mass: Array
        if atomic_mass is not None:
            self.mass = atomic_mass.reshape(-1, 1)
        else:
            logger.warning("Extracting atomic masses from input structure")
            self.mass = self.structure.mass.reshape(-1, 1)

        self.velocity: Array
        if initial_velocity is not None:
            self.velocity = initial_velocity
        else:
            assert (
                target_temperature is not None
            ), "At least one of temperature or initial velocity must be given"
            logger.warning(
                f"Generating random velocities ({target_temperature:0.2f} K)")
            self.velocity = self.generate_random_velocity(
                target_temperature, random_seed
            )

        self.thermostat_time_constant: float
        if thermostat_time_constant is not None:
            self.thermostat_time_constant = thermostat_time_constant
        else:
            self.thermostat_time_constant = 100 * self.time_step

        self.step: int = 0
        self.elapsed_time: float = 0.0
        self.temperature: Array = _get_temperature(self.velocity, self.mass)

    def run_simulation(
        self,
        num_steps: int = 1,
        print_freq: Optional[int] = None,
    ) -> None:
        """Run molecular simulation for a given number of steps."""
        if print_freq is None:
            print_freq = 1 if num_steps < 100 else int(0.01 * num_steps)
        try:
            for _ in range(num_steps):
                if (print_freq > 0) and (self.step % print_freq == 0):
                    print(self._repr_params())
                self.molecular_dynamics_step()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")

    def molecular_dynamics_step(self) -> None:
        """Update parameters for next time step."""
        self.verlet_integration()
        self.step += 1
        self.elapsed_time += self.time_step
        self.temperature = _get_temperature(self.velocity, self.mass)
        if self.target_temperature is not None:
            self.brendsen_thermostat(self.target_temperature)

    def verlet_integration(self) -> None:
        """Update atom positions and velocities based on Verlet algorithm."""
        new_position = (
            self.position
            + self.velocity * self.time_step
            + 0.5 * self.force * self.time_step**2
        )
        self.structure = replace(self.structure, position=new_position)
        new_force = _compute_force(self.potential, self.structure)
        self.velocity += 0.5 * (self.force + new_force) * self.time_step
        self.force = new_force

    def brendsen_thermostat(self, target_temperature: float) -> None:
        """Control simulation temperature using Brendsen algorithm."""
        scaling_factor = jnp.sqrt(
            1.0
            + (self.time_step / self.thermostat_time_constant)
            * (self.temperature / target_temperature - 1.0)
        )
        self.velocity /= scaling_factor

    def generate_random_velocity(self, temperature: float, seed: int) -> Array:
        """Generate velocities with Maxwell-Boltzmann distribution."""
        key = jax.random.PRNGKey(seed)
        velocity = jax.random.normal(key, shape=(self.structure.natoms, 3))
        temperature = _get_temperature(velocity, self.mass)
        velocity *= jnp.sqrt(cast(Array, self.target_temperature) / temperature)
        velocity -= _get_center_of_mass(velocity, self.mass)
        return velocity

    def _repr_params(self) -> str:
        return (
            f"{self.step:<10} "
            f"time:{self.elapsed_time:<15.10f} "
            f"Temp:{self.temperature:<15.10f} "
            f"Etot:{self.get_total_energy():<15.10f} "
            f"Epot={self.get_potential_energy():<15.10f} "
        )

    @property
    def position(self) -> Array:
        return self.structure.position

    @property
    def natoms(self) -> int:
        return self.structure.natoms

    @property
    def elements(self) -> Tuple[Element]:
        return self.structure.elements

    def get_com_velocity(self) -> Array:
        """Return center of mass velocity."""
        return _get_center_of_mass(self.velocity, self.mass)

    def get_com_position(self) -> Array:
        """Return center of mass position."""
        return _get_center_of_mass(self.position, self.mass)

    def get_potential_energy(self) -> Array:
        return _get_potential_energy(self.potential, self.structure)

    def get_kinetic_energy(self) -> Array:
        return _get_kinetic_energy(self.velocity, self.mass)

    def get_total_energy(self) -> Array:
        return self.get_potential_energy() + self.get_kinetic_energy()
