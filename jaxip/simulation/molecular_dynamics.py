from typing import Optional, Protocol, Tuple

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



class MDSimulator:
    """A simple molecular dynamics simulator for a given potential function."""

    def __init__(
        self,
        potential: PotentialInterface,
        initial_structure: Structure,
        time_step: float,
        temperature: Optional[float] = None,
        velocity: Optional[Array] = None,
        mass: Optional[Array] = None,
        random_seed: int = 12345,
    ):
        self.potential = potential
        self.structure = initial_structure
        self.time_step = time_step
        self.temperature = temperature
        self.force: Array
        self.mass: Array

        self.force = self._compute_force(self.structure)

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

        if velocity is not None:
            self.velocity = velocity
        else:
            assert (
                temperature is not None
            ), "At least one of temperature or initial velocity must be given"
            self.velocity = self._generate_random_velocity(temperature, random_seed)

    def run_simulation(self, steps: int = 1) -> None:
        for step in range(steps):
            print(f"{step=}, total_energy={self.potential(self.structure)}")
            if self.temperature is not None:
                self.use_thermostat()
            self.verlet_integration()

    def verlet_integration(self) -> None:
        self.structure.position += (
            self.velocity * self.time_step + 0.5 * self.force * self.time_step**2
        )
        new_force = self._compute_force(self.structure)
        self.velocity += 0.5 * (self.force + new_force) * self.time_step
        self.force = new_force
        print(self.position)

    def use_thermostat(self) -> None:
        kinetic_energy = 0.5 * jnp.sum(self.velocity**2)
        current_temperature = 2 * kinetic_energy / (3 * self.natoms)
        scaling_factor = jnp.sqrt(self.temperature / current_temperature)
        self.velocity *= scaling_factor

    def _generate_random_velocity(self, temperature: float, seed: int) -> Array:
        key = jax.random.PRNGKey(seed)
        # KB = 1.38064852e-23  # Boltzmann constant in J/K
        # std_dev = np.sqrt(KB * temperature / self.mass)
        velocity = jax.random.normal(key, shape=(self.structure.natoms, 3))
        center_of_mass_velocity = jnp.sum(self.mass * velocity, axis=0) / jnp.sum(self.mass)
        velocity -= center_of_mass_velocity
        return velocity

    def _compute_force(self, structure: Structure) -> Array:
        return self.potential.compute_force(structure)

    @property
    def position(self) -> Array:
        return self.structure.position

    @property
    def natoms(self) -> int:
        return self.structure.natoms

    @property
    def elements(self) -> Tuple[Element]:
        return self.structure.elements
