from jaxip.types import Array
from jaxip.atoms import Structure
from typing import Protocol, Optional


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
        temperature: Optional[float],
    ):
        self.potential = potential
        self.structure = initial_structure
        self.time_step = time_step
        self.temperature = temperature

    # def verlet_integration(self) -> None:
    #     forces = self.potential.compute_forces(self.structure)
    #     self.structure += self.velocity * self.time_step + 0.5 * forces * self.time_step**2
    #     new_forces = self.potential_function.compute_forces(self.structure)
    #     self.velocity += 0.5 * (forces + new_forces) * self.time_step
    #
    # def apply_thermostat(self) -> None:
    #     kinetic_energy = 0.5 * np.sum(self.velocity**2)
    #     current_temperature = 2 * kinetic_energy / (3 * len(self.structure))
    #     scaling_factor = np.sqrt(self.temperature / current_temperature)
    #     self.velocity *= scaling_factor
    #
    # def run_simulation(self, num_steps: int) -> None:
    #     for step in range(num_steps):
    #         if self.temperature is not None:
    #             self.apply_thermostat()
    #         self.verlet_integration()
