from typing import Protocol

import jax.numpy as jnp

from pantea.types import Array


class MDSimulatorInterface(Protocol):
    time_step: float


class SystemInterface(Protocol):
    velocities: Array

    def get_temperature(self) -> Array:
        ...


class BrendsenThermostat:
    """Control simulation temperature using Brendsen algorithm."""

    def __init__(
        self,
        target_temperature: float,
        time_constant: float,
    ) -> None:
        self.target_temperature = target_temperature
        self.time_constant = time_constant

    def get_rescaled_velocities(
        self,
        simulator: MDSimulatorInterface,
        system: SystemInterface,
    ) -> Array:
        scaling_factor = jnp.sqrt(
            1.0
            + (simulator.time_step / self.time_constant)
            * (system.get_temperature() / self.target_temperature - 1.0)
        )
        return system.velocities / scaling_factor
