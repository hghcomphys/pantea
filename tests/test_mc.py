import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

from typing import Tuple

import jax.numpy as jnp
import pytest
from ase import Atoms

from pantea.atoms.structure import Structure
from pantea.simulation import LJPotential, MCSimulator
from pantea.simulation.system import System
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


class TestMCSimulator:
    mc = MCSimulator(
        translate_step=0.3 * units.FROM_ANGSTROM,
        target_temperature=300.0,
        movements_per_step=10,
    )
    sys = System.from_structure(
        structure=get_structure(),
        potential=get_potential(),
        temperature=300.0,
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
        assert jnp.allclose(mc.translate_step, expected[1])

    @pytest.mark.parametrize(
        "mc, sys, expected",
        [
            (
                mc,
                sys,
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
        sys: System,
        expected: Tuple,
    ) -> None:
        mc.simulate_one_step(sys)
        assert mc.step == 1
        assert jnp.allclose(mc.target_temperature, expected[0])
        assert jnp.allclose(mc.translate_step, expected[1])
        assert jnp.allclose(sys.get_potential_energy(), expected[2])
