from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import jax
import jax.numpy as jnp

from pantea.atoms import Structure
from pantea.atoms.box import Box
from pantea.atoms.element import ElementMap
from pantea.logger import logger
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
    velocities: Array,
    masses: Array,
    positions: Array,
    forces: Array,
) -> Array:
    return 2 * _get_kinetic_energy(velocities, masses) + jnp.sum(positions * forces)


@jax.jit
def _calculate_center_of_mass(array: Array, masses: Array) -> Array:
    return jnp.sum(masses * array, axis=0) / jnp.sum(masses)


class PotentialInterface(Protocol):
    def __call__(self, structure: Structure) -> Array:
        ...

    def compute_forces(self, structure: Structure) -> Array:
        ...


@dataclass
class System:
    """An extended atom Structure for molecular simulations (e.g. MD)."""

    potential: PotentialInterface
    structure: Structure
    velocities: Array
    masses: Array

    @classmethod
    def from_structure(
        cls,
        structure: Structure,
        potential: PotentialInterface,
        temperature: float = 300.0,
        seed: int = 2024,
    ) -> System:
        logger.debug(f"Creating {cls.__name__} from Structure")
        masses = ElementMap.get_masses_from_structure(structure).reshape(-1, 1)
        velocities = cls.generate_random_velocities(
            jnp.array(temperature), masses, seed
        )
        return cls(potential, deepcopy(structure), velocities, masses)

    def __post_init__(self) -> None:
        self.update_forces_from_positions()
        self.update_total_potential_energy_from_positions()

    @classmethod
    def generate_random_velocities(
        cls,
        temperature: Array,
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

    def update_forces_from_positions(self) -> None:
        self.structure.forces = self.potential.compute_forces(self.structure)

    def update_total_potential_energy_from_positions(self) -> None:
        self.structure.total_energy = self.potential(self.structure)

    @classmethod
    def compute_forces(
        cls,
        potential: PotentialInterface,
        structure: Structure,
    ) -> Array:
        return potential.compute_forces(structure)

    def get_elements(self) -> Tuple[Element, ...]:
        return self.structure.get_elements()

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
        """Return total potential energy."""
        return self.potential(self.structure)

    def get_kinetic_energy(self) -> Array:
        return _get_kinetic_energy(self.velocities, self.masses)

    def get_total_energy(self) -> Array:
        return self.get_potential_energy() + self.get_kinetic_energy()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"potential={self.potential.__class__.__name__}, "
            f"structure={self.structure}, "
            f"temperature={self.get_temperature():.2f})"
        )

    @property
    def positions(self) -> Array:
        return self.structure.positions

    @property
    def forces(self) -> Array:
        return self.structure.forces

    @property
    def box(self) -> Optional[Box]:
        return self.structure.box

    @property
    def natoms(self) -> int:
        return self.structure.natoms
