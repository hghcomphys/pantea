import os
from typing import Tuple

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
import pytest

from jaxip.atoms.structure import Structure
from jaxip.descriptors.acsf import ACSF, G2, G3, CutoffFunction
from jaxip.descriptors.acsf.symmetry import EnvironmentElements
from jaxip.types import Array


# @pytest.fixture
def lj_acsf() -> ACSF:
    """Initialize using the `add` method."""
    acsf: ACSF = ACSF("Ne")
    cfn: CutoffFunction = CutoffFunction.from_cutoff_type(
        r_cutoff=3.0, cutoff_type="tanhu"
    )
    acsf.add(G2(cfn, eta=1.00, r_shift=0.00), "Ne")
    acsf.add(G2(cfn, eta=1.00, r_shift=0.25), "Ne")
    acsf.add(G2(cfn, eta=1.00, r_shift=0.50), "Ne")
    acsf.add(G2(cfn, eta=1.00, r_shift=0.75), "Ne")
    acsf.add(G2(cfn, eta=1.00, r_shift=1.00), "Ne")
    return acsf


def h2o_acsf() -> ACSF:
    """Initialize directly from the radial and angular terms."""
    return ACSF(
        central_element="O",
        radial_symmetry_functions=(
            (
                EnvironmentElements(central="O", neighbor_j="H"),
                G2(
                    cfn=CutoffFunction.from_cutoff_type(
                        r_cutoff=5.9043202, cutoff_type="tanhu"
                    ),  # r_cutoff = box.length / 2
                    r_shift=0.0,
                    eta=0.001,
                ),
            ),
        ),
        angular_symmetry_functions=(
            (
                EnvironmentElements(
                    central="O", neighbor_j="H", neighbor_k="H"
                ),
                G3(
                    cfn=CutoffFunction.from_cutoff_type(
                        r_cutoff=5.9043202, cutoff_type="tanhu"
                    ),  # r_cutoff = box.length / 2
                    eta=0.07,
                    zeta=1.0,
                    lambda0=1.0,
                    r_shift=0.0,
                ),
            ),
        ),
    )


class TestACSF:
    lj_structure: Structure = Structure.from_dict(
        {
            "positions": [
                [0.0, 0.0, 0.0],
                [0.588897275, 0.588897275, 0.588897275],
            ],
            "elements": ["Ne", "Ne"],
        }
    )
    h2o_structure: Structure = Structure.from_dict(
        {
            "lattice": [
                [11.8086403654, 0.0, 0.0],
                [0.0, 11.8086403654, 0.0],
                [0.0, 0.0, 11.8086403654],
            ],
            "positions": [
                [0.2958498542, -0.8444146738, 1.9618569793],
                [1.7226932399, -1.8170359274, 0.5237867306],
                [1.2660050151, -4.3958431356, 0.822408813],
                [2.5830855482, 6.6971705174, 2.042321518],
                [6.3733092527, 4.0651032678, 1.6571254123],
                [4.5893132971, 4.2984466507, 3.33215409],
                [-2.2136818837, 1.3673642557, 2.7768013741],
                [-1.9466635812, -0.3397746484, 3.3828743394],
                [-0.4559776878, 0.1681476423, -5.430449296],
                [-1.2831542798, 4.5473991714, -2.294184217],
                [-2.3516507887, 0.9376745482, -4.0756290424],
                [-1.7819663898, 4.1957022409, -4.0621741923],
            ],
            "elements": [
                "O",
                "H",
                "H",
                "O",
                "H",
                "H",
                "O",
                "H",
                "H",
                "O",
                "H",
                "H",
            ],
        }
    )

    @pytest.mark.parametrize(
        "acsf, expected",
        [
            (
                lj_acsf(),
                ("Ne", 5, 0, 5, 3.0),
            ),
        ],
    )
    def test_acsf_attributes(
        self,
        acsf: ACSF,
        expected: Tuple,
    ) -> None:
        assert acsf.central_element == expected[0]
        assert acsf.num_radial_symmetry_functions == expected[1]
        assert acsf.num_angular_symmetry_functions == expected[2]
        assert acsf.num_symmetry_functions == expected[3]
        assert acsf.r_cutoff == expected[4]

    @pytest.mark.parametrize(
        "acsf, structure, expected",
        [
            (
                lj_acsf(),
                lj_structure,
                jnp.tile(
                    jnp.asarray(
                        [
                            0.0683537673,
                            0.1069323809,
                            0.1476281601,
                            0.1798632214,
                            0.1933876418,
                        ]
                    ),
                    (2, 1),
                ),
            ),
        ],
    )
    def test_acsf_without_pbc(
        self,
        acsf: ACSF,
        structure: Structure,
        expected: Array,
    ) -> None:
        assert acsf(structure).shape == expected.shape
        assert jnp.allclose(acsf(structure), expected)

    @pytest.mark.parametrize(
        "acsf, structure, expected",
        [
            (
                h2o_acsf(),
                h2o_structure,
                (
                    (4, 2),
                    jnp.asarray([[0.3333029149, 0.0002204830]]),
                ),
            ),
        ],
    )
    def test_acsf_with_pbc(
        self,
        acsf: ACSF,
        structure: Structure,
        expected: Array,
    ) -> None:
        assert acsf(structure).shape == expected[0]
        assert jnp.allclose(
            acsf(structure, atom_indices=jnp.asarray(0)), expected[1]
        )
