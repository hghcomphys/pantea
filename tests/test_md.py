import os
from typing import Tuple

import jax.numpy as jnp
import pytest
from ase import Atoms

from jaxip.atoms.structure import Structure
from jaxip.simulation.lj import LJPotential
from jaxip.simulation.md import MDSimulator
from jaxip.units import units

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"


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
        potential=get_potential(),
        initial_structure=get_structure(),
        time_step=0.5 * units.FROM_FEMTO_SECOND,
        temperature=300.0,
    )

    @pytest.mark.parametrize(
        "md, expected",
        [
            (
                md,
                (
                    0.5 * units.FROM_FEMTO_SECOND,
                    262.15531167,
                    5.70328776e-07,
                    0.00995778,
                    jnp.asarray([11.33835602, 11.33835602, 11.33835602]),
                ),
            ),
        ],
    )
    def test_general_attributes(
        self,
        md: MDSimulator,
        expected: Tuple,
    ) -> None:
        assert md.step == 0
        assert jnp.allclose(md.elapsed_time, 0.0)
        assert jnp.allclose(md.time_step, expected[0])
        assert jnp.allclose(md.get_center_of_mass_velocity(), jnp.zeros(3))
        assert jnp.allclose(md.temperature, expected[1])
        assert jnp.allclose(md.get_pressure(), expected[2])
        assert jnp.allclose(md.get_total_energy(), expected[3])
        assert jnp.allclose(md.get_center_of_mass_position(), expected[4])

    @pytest.mark.parametrize(
        "md, structure",
        [
            (
                md,
                get_structure(),
            ),
        ],
    )
    def test_structure_attributes(
        self,
        md: MDSimulator,
        structure: Structure,
    ) -> None:
        assert jnp.allclose(md.positions, structure.positions)
        assert jnp.allclose(md.masses, structure.get_masses())

    @pytest.mark.parametrize(
        "md, expected",
        [
            (
                md,
                (
                    0.5 * units.FROM_FEMTO_SECOND,
                    261.23330876,
                    5.68325481e-07,
                    0.00992273,
                    jnp.asarray([11.33835602, 11.33835602, 11.33835602]),
                ),
            ),
        ],
    )
    def test_update(
        self,
        md: MDSimulator,
        expected: Tuple,
    ) -> None:
        md.update()
        assert md.step == 1
        assert jnp.allclose(md.elapsed_time, expected[0])
        assert jnp.allclose(md.time_step, expected[0])
        assert jnp.allclose(md.get_center_of_mass_velocity(), jnp.zeros(3))
        assert jnp.allclose(md.get_center_of_mass_position(), expected[4])
