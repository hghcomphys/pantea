from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from pantea.atoms.box import _shift_inside_box
from pantea.logger import logger
from pantea.simulation.system import System
from pantea.simulation.thermostat import BrendsenThermostat
from pantea.types import Array
from pantea.units import units


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
            to the target value, defaults to None
        :type thermostat: Optional[BrendsenThermostat], optional

        This simulator is intended to be used for small systems of atoms during
        the potential training in order to generate new training dataset.
        """

        logger.debug(f"Initializing {self.__class__.__name__}")
        self.time_step: Array = jnp.array(time_step)
        self.thermostat = thermostat
        self.step: int = 0
        self.elapsed_time: float = 0.0

    def simulate_one_step(self, system: System) -> None:
        """Update parameters for next time step."""
        self.verlet_integration(system)
        self.step += 1
        self.elapsed_time += float(self.time_step)
        if self.thermostat is not None:
            system.velocities = self.thermostat.get_rescaled_velocities(self, system)

    def verlet_integration(self, system: System) -> None:
        """Update atom positions, velocities, and forces based on Verlet algorithm."""
        new_positions = _get_verlet_new_positions(
            system.positions, system.velocities, system.forces, self.time_step
        )
        if system.box is not None:
            new_positions = _shift_inside_box(new_positions, system.box.lattice)
        system.structure.positions = new_positions
        new_forces = system.potential.compute_forces(system.structure)
        system.velocities = _get_verlet_new_velocities(
            system.velocities, system.forces, new_forces, self.time_step
        )
        system.structure.forces = new_forces

    def repr_physical_params(self, system: System) -> str:
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
