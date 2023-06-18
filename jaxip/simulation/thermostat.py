from jaxip.types import Array
import jax.numpy as jnp
from typing import Protocol


class MDSimulatorInterface(Protocol):
    temperature: float
    time_step: float
    velocity: Array


class BrendsenThermostat:
    """Control simulation temperature using Brendsen algorithm."""

    def __init__(
        self,
        target_temperature: float,
        time_constant: float,
    ) -> None:
        self.target_temperature = target_temperature
        self.time_constant = time_constant

    def get_rescaled_velocity(
        self,
        simulator: MDSimulatorInterface,
    ) -> Array:
        scaling_factor = jnp.sqrt(
            1.0
            + (simulator.time_step / self.time_constant)
            * (simulator.temperature / self.target_temperature - 1.0)
        )
        return simulator.velocity / scaling_factor
