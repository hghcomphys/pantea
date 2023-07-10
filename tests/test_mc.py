import os
from typing import Tuple

import jax.numpy as jnp
import pytest
from ase import Atoms

from jaxip.atoms.structure import Structure
from jaxip.simulation.lj import LJPotential
from jaxip.simulation.mc import MCSimulator
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


class TestMCSimulator:
    mc = MCSimulator(
        potential=get_potential(),
        initial_structure=get_structure(),
        temperature=300,
        translate_step=0.3 * units.FROM_ANGSTROM,
        movements_per_step=10,
    )

    @pytest.mark.parametrize(
        "mc, expected",
        [
            (
                mc,
                (
                    300.0,
                    0.3 * units.FROM_ANGSTROM,
                    -4.575687e-06,
                ),
            ),
        ],
    )
    def test_general_attributes(
        self,
        mc: MCSimulator,
        expected: Tuple,
    ) -> None:
        assert mc.step == 0
        assert jnp.allclose(mc.temperature, expected[0])
        assert jnp.allclose(mc.translate_step, expected[1])
        assert jnp.allclose(mc.energy, expected[2])

    @pytest.mark.parametrize(
        "mc, structure",
        [
            (
                mc,
                get_structure(),
            ),
        ],
    )
    def test_structure_attributes(
        self,
        mc: MCSimulator,
        structure: Structure,
    ) -> None:
        assert jnp.allclose(mc.positions, structure.positions)
        assert jnp.allclose(mc.masses, structure.get_masses())

    @pytest.mark.parametrize(
        "mc, expected",
        [
            (
                mc,
                (
                    300.0,
                    0.3 * units.FROM_ANGSTROM,
                    -5.7142543e-06,
                ),
            ),
        ],
    )
    def test_update(
        self,
        mc: MCSimulator,
        expected: Tuple,
    ) -> None:
        mc.update()
        assert mc.step == 1
        assert jnp.allclose(mc.temperature, expected[0])
        assert jnp.allclose(mc.translate_step, expected[1])
        assert jnp.allclose(mc.energy, expected[2])
