from __future__ import annotations

from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp

from pantea.simulation.system import _get_temperature
from pantea.types import Array


@jax.jit
def _get_rescaled_velocities(
    params: BrendsenThermostatParams,
    velocities: Array,
):
    scaling_factor = 1.0 / jnp.sqrt(
        1.0
        + (params.time_step / params.time_constant)
        * (params.current_temperature / params.target_temperature - 1.0)
    )
    return velocities * scaling_factor


class MDSimulatorInterface(Protocol):
    time_step: float


class SystemInterface(Protocol):
    velocities: Array

    def get_temperature(self) -> Array:
        ...


class BrendsenThermostatParams(NamedTuple):
    time_step: Array
    time_constant: Array
    current_temperature: Array
    target_temperature: Array


class BrendsenThermostat:
    """Control simulation temperature using Brendsen thermostat."""

    def __init__(
        self,
        target_temperature: float,
        time_constant: float,
    ) -> None:
        self.target_temperature: Array = jnp.array(target_temperature)
        self.time_constant: Array = jnp.array(time_constant)

    def get_rescaled_velocities(
        self,
        simulator: MDSimulatorInterface,
        system: SystemInterface,
    ) -> Array:
        current_temperature = system.get_temperature()
        params = BrendsenThermostatParams(
            simulator.time_step,
            self.time_constant,
            current_temperature,
            self.target_temperature,
        )
        return _get_rescaled_velocities(params, system.velocities)
