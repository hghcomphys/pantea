from dataclasses import replace
from typing import Iterator, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp

from jaxip.atoms import Structure
from jaxip.atoms.element import ElementMap
from jaxip.types import Array, Element


class PotentialInterface(Protocol):
    def __call__(self, structure: Structure) -> Array:
        ...

    def compute_force(self, structure: Structure) -> Array:
        ...


@jax.jit
def _compute_com_velocity(velocity: Array, mass: Array) -> Array:
    return jnp.sum(mass * velocity, axis=0) / jnp.sum(mass)


@jax.jit
def _compute_kinetic_energy(velocity: Array, mass: Array) -> Array:
    return 0.5 * jnp.sum(mass * velocity**2)


@jax.jit
def _compute_temprature(velocity: Array, mass: Array) -> Array:
    kinetic_energy = _compute_kinetic_energy(velocity, mass)
    natoms = velocity.shape[0]
    return 2 * kinetic_energy / (3 * natoms)


def _compute_force(
    potential: PotentialInterface,
    structure: Structure,
) -> Array:
    return potential.compute_force(structure)


def _compute_energy(
    potential: PotentialInterface,
    structure: Structure,
) -> Array:
    return potential(structure)


class MDSimulator:
    """A simple molecular dynamics simulator for a given potential function."""

    def __init__(
        self,
        potential: PotentialInterface,
        initial_structure: Structure,
        time_step: float,
        temperature: Optional[float] = None,
        initial_velocity: Optional[Array] = None,
        mass: Optional[Array] = None,
        random_seed: int = 12345,
    ):
        self.potential = potential
        self.structure = initial_structure
        self.time_step = time_step
        self.temperature = temperature
        self.force = _compute_force(self.potential, self.structure)
        self.step: int = 0

        self.mass: Array
        if mass is not None:
            self.mass = mass
        else:
            to_element = self.structure.element_map.atom_type_to_element
            elements = (to_element[int(at)] for at in self.structure.atom_type)
            self.mass = jnp.array(
                tuple(
                    ElementMap.element_to_atomic_mass(element) for element in elements
                )
            ).reshape(-1, 1)

        self.velocity: Array
        if initial_velocity is not None:
            self.velocity = initial_velocity
        else:
            assert (
                temperature is not None
            ), "At least one of temperature or initial velocity must be given"
            self.velocity = self._generate_random_velocity(temperature, random_seed)

    def run_simulation(self, num_steps: int = 1, print_freq: int = 10) -> None:
        for _ in range(num_steps):
            if self.step % print_freq == 0:
                print(next(self._get_physical_params()))
            if self.temperature is not None:
                self.apply_thermostat()
            self.verlet_integration()

    def _get_physical_params(self) -> Iterator[str]:
        potential_energy = _compute_energy(self.potential, self.structure)
        kinitic_energy = _compute_kinetic_energy(self.velocity, self.mass)
        total_energy = potential_energy + kinitic_energy
        temperature = _compute_temprature(self.velocity, self.mass)
        output = (
            f"{self.step:<10}"
            f"Time:{self.time_step * self.step:<15.10f} "
            f"Temp:{temperature:<15.10f} "
            f"Etot:{total_energy:<15.10f} "
            # f"Epot={potential_energy:<15.10f} "
        )
        yield output

    def verlet_integration(self) -> None:
        new_position = (
            self.position
            + self.velocity * self.time_step
            + 0.5 * self.force * self.time_step**2
        )
        self.structure = replace(self.structure, position=new_position)
        new_force = _compute_force(self.potential, self.structure)
        self.velocity += 0.5 * (self.force + new_force) * self.time_step
        self.force = new_force
        self.step += 1

    def apply_thermostat(self) -> None:
        current_temperature = _compute_temprature(self.velocity, self.mass)
        scaling_factor = jnp.sqrt(self.temperature / current_temperature)
        self.velocity *= scaling_factor

    def _generate_random_velocity(self, temperature: float, seed: int) -> Array:
        key = jax.random.PRNGKey(seed)
        KB = 3.166811563e-6  # Boltzmann constant in Hartree/K
        std_dev = jnp.sqrt(KB * temperature / self.mass)
        velocity = std_dev * jax.random.normal(key, shape=(self.structure.natoms, 3))
        velocity -= _compute_com_velocity(velocity, self.mass)
        return velocity

    @property
    def position(self) -> Array:
        return self.structure.position

    @property
    def natoms(self) -> int:
        return self.structure.natoms

    @property
    def elements(self) -> Tuple[Element]:
        return self.structure.elements
