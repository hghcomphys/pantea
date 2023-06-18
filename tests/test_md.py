import os
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import pytest
from ase import Atoms

from jaxip.atoms.structure import Structure
from jaxip.simulation.md import MDSimulator
from jaxip.types import Array
from jaxip.units import units

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"


def get_structure() -> Structure:
    d = 6  # Angstrom
    uc = Atoms("He", positions=[(d / 2, d / 2, d / 2)], cell=(d, d, d))
    s = Structure.create_from_ase(uc.repeat((2, 2, 2)))
    return s


def get_potential():
    @partial(jax.jit, static_argnums=0)
    def _compute_pair_energy(obj, r: Array) -> Array:
        term = obj.sigma / r
        term6 = term**6
        return 4.0 * obj.epsilon * term6 * (term6 - 1.0)

    @partial(jax.jit, static_argnums=0)
    def _compute_pair_force(obj, r: Array, R: Array) -> Array:
        term = obj.sigma / r
        term6 = term**6
        force_factor = -24.0 * obj.epsilon / (r * r) * term6 * (2.0 * term6 - 1.0)
        return jnp.expand_dims(force_factor, axis=-1) * R

    class LJPotential:
        def __init__(
            self,
            sigma: float,
            epsilon: float,
            r_cutoff: float,
        ) -> None:
            self.sigma = sigma
            self.epsilon = epsilon
            self.r_cutoff = r_cutoff

        def __call__(self, structure: Structure) -> Array:
            r, _ = structure.calculate_distance(atom_index=jnp.arange(structure.natoms))
            mask = (0 < r) & (r < self.r_cutoff)
            pair_energies = _compute_pair_energy(self, r)
            return 0.5 * jnp.where(mask, pair_energies, 0.0).sum()  # type: ignore

        def compute_force(self, structure: Structure) -> Array:
            r, R = structure.calculate_distance(atom_index=jnp.arange(structure.natoms))
            mask = (0 < r) & (r < self.r_cutoff)
            pair_forces = jnp.where(
                jnp.expand_dims(mask, axis=-1),
                _compute_pair_force(self, r, R),
                jnp.zeros_like(R),
            )
            return jnp.sum(pair_forces, axis=1)  # type: ignore

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
        temperature=300,
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
        assert jnp.allclose(md.get_com_velocity(), jnp.zeros(3))
        assert jnp.allclose(md.temperature, expected[1])
        assert jnp.allclose(md.get_pressure(), expected[2])
        assert jnp.allclose(md.get_total_energy(), expected[3])
        assert jnp.allclose(md.get_com_position(), expected[4])

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
        assert jnp.allclose(md.position, structure.position)
        assert jnp.allclose(md.mass, structure.mass)

    @pytest.mark.parametrize(
        "md, expected",
        [
            (
                md,
                (
                    0.5 * units.FROM_FEMTO_SECOND,
                    260.89277174,
                    5.68325481e-07,
                    0.00992273,
                    jnp.asarray([11.33835602, 11.33835602, 11.33835602]),
                ),
            ),
        ],
    )
    def test_md_step(
        self,
        md: MDSimulator,
        expected: Tuple,
    ) -> None:
        md.molecular_dynamics_step()
        assert md.step == 1
        assert jnp.allclose(md.elapsed_time, expected[0])
        assert jnp.allclose(md.time_step, expected[0])
        assert jnp.allclose(md.get_com_velocity(), jnp.zeros(3))
        assert jnp.allclose(md.temperature, expected[1])
        assert jnp.allclose(md.get_pressure(), expected[2])
        assert jnp.allclose(md.get_total_energy(), expected[3])
        assert jnp.allclose(md.get_com_position(), expected[4])
