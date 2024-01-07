import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

from typing import Tuple

import jax.numpy as jnp
import pytest
from ase import Atoms

from pantea.atoms.element import ElementMap
from pantea.atoms.structure import Structure
from pantea.simulation import LJPotential, MDSimulator
from pantea.simulation.system import System
from pantea.simulation.thermostat import BrendsenThermostat
from pantea.units import units


def get_structure() -> Structure:
    d = 6  # Angstrom
    uc = Atoms("He", positions=[(d / 2, d / 2, d / 2)], cell=(d, d, d))
    s = Structure.from_ase(uc.repeat((2, 2, 2)))
    return s


def get_potential() -> LJPotential:
    return LJPotential(
        sigma=2.5238 * units.FROM_ANGSTROM,
        epsilon=4.7093e-04 * units.FROM_ELECTRON_VOLT,
        r_cutoff=6.3095 * units.FROM_ANGSTROM,
    )


class TestMDSimulator:
    md = MDSimulator(
        time_step=0.5 * units.FROM_FEMTO_SECOND,
        thermostat=BrendsenThermostat(
            target_temperature=300.0,
            time_constant=100 * 0.5 * units.FROM_FEMTO_SECOND,
        ),
    )
    sys = System.from_structure(
        structure=get_structure(),
        potential=get_potential(),
        temperature=300.0,
        seed=2023,
    )

    @pytest.mark.parametrize(
        "md, sys, expected",
        [
            (
                md,
                sys,
                (
                    0.5 * units.FROM_FEMTO_SECOND,
                    256.72903,
                    5.70328776e-07,
                    0.00975157,
                    jnp.array([11.33835602, 11.33835602, 11.33835602]),
                ),
            ),
        ],
    )
    def test_general_attributes(
        self,
        md: MDSimulator,
        sys: System,
        expected: Tuple,
    ) -> None:
        assert md.step == 0
        assert jnp.allclose(md.elapsed_time, 0.0)
        assert jnp.allclose(md.time_step, expected[0])
        assert jnp.allclose(sys.get_center_of_mass_velocity(), jnp.zeros(3))
        assert jnp.allclose(sys.get_center_of_mass_position(), expected[4])
        # assert jnp.allclose(sys.get_temperature(), expected[1])
        # assert jnp.allclose(sys.get_pressure(), expected[2])
        # assert jnp.allclose(sys.get_total_energy(), expected[3])

    @pytest.mark.parametrize(
        "md, sys, structure",
        [
            (
                md,
                sys,
                get_structure(),
            ),
        ],
    )
    def test_structure_attributes(
        self,
        md: MDSimulator,
        sys: System,
        structure: Structure,
    ) -> None:
        assert jnp.allclose(sys.positions, structure.positions)
        assert jnp.allclose(sys.masses, ElementMap.get_masses_from_structure(structure))

    @pytest.mark.parametrize(
        "md, sys, expected",
        [
            (
                md,
                sys,
                (
                    0.5 * units.FROM_FEMTO_SECOND,
                    261.23330876,
                    5.68325481e-07,
                    0.00992273,
                    jnp.array([11.33835602, 11.33835602, 11.33835602]),
                ),
            ),
        ],
    )
    def test_update(
        self,
        md: MDSimulator,
        sys: System,
        expected: Tuple,
    ) -> None:
        md.simulate_one_step(sys)
        assert md.step == 1
        assert jnp.allclose(md.elapsed_time, expected[0])
        assert jnp.allclose(md.time_step, expected[0])
        assert jnp.allclose(sys.get_center_of_mass_velocity(), jnp.zeros(3))
        assert jnp.allclose(sys.get_center_of_mass_position(), expected[4])
