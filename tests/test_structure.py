import os
from typing import Any, Dict, Tuple

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax.numpy as jnp
import pytest

from jaxip.structure.structure import Structure
from jaxip.types import dtype as _dtype

LJ_DATA: Dict[str, Any] = {
    "position": [[0.0, 0.0, 0.0], [0.583123772, 0.583123772, 0.583123772]],
    "element": ["Ne", "Ne"],
    "charge": [0.0, 0.0],
    "energy": [0.0, 0.0],
    "force": [
        [-11.4260918, -11.4260918, -11.4260918],
        [11.4260918, 11.4260918, 11.4260918],
    ],
    "total_energy": [-0.21838404],
    "comment": ["r = 1.01000000E+00, E = -2.18384040E-01, dEdr = -1.97905715E+01"],
    "total_charge": [],
    "lattice": [],
    "atom_type": [1, 1],
}

H2O_DATA: Dict[str, Any] = {
    "lattice": [
        [11.8086403654, 0.0, 0.0],
        [0.0, 11.8086403654, 0.0],
        [0.0, 0.0, 11.8086403654],
    ],
    "position": [
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
    "element": ["O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H"],
    "charge": [
        0.423192,
        -0.243872,
        -0.313417,
        0.439248,
        -0.136408,
        -0.154832,
        0.418857,
        -0.401243,
        -0.0766178,
        0.411994,
        -0.0691934,
        -0.297707,
    ],
    "energy": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    "force": [
        [-0.4727795521, -0.0084561951, 0.0460983059],
        [0.1920381912, 0.0587784661, 0.6142110612],
        [-0.8869675228, -0.7832124482, -0.7448845423],
        [0.7405673293, 0.4095246028, 0.1627052569],
        [-0.7272286968, 0.2123232582, 0.5568660265],
        [0.6038769757, -0.0797700393, -0.5249089268],
        [-0.2617806281, -0.2369410958, 0.1869405738],
        [1.2121859011, 0.4399940142, -0.0045549514],
        [-0.1108903386, -0.3317914324, -0.6560763592],
        [-0.1172658121, -0.1885000212, -0.5958629059],
        [-0.1934543148, 0.5415107499, 0.4505828542],
        [0.0216973013, -0.0334599566, 0.508882732],
    ],
    "total_energy": [-32.1390027258],
    "total_charge": [8.0e-07],
    "atom_type": [2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1],
}


class TestStructure:

    lj: Structure = Structure.create_from_dict(LJ_DATA, dtype=jnp.float32)  # type: ignore
    h2o: Structure = Structure.create_from_dict(H2O_DATA, r_cutoff=11.0)
    atom_attributes: Tuple[str, ...] = tuple(
        ["position", "force", "energy", "total_energy", "charge", "total_charge"]
    )

    @pytest.mark.parametrize(
        "structure, expected",
        [
            (
                lj,
                tuple(
                    jnp.asarray(LJ_DATA[attr])
                    for attr in Structure._get_atom_attributes()
                ),
            ),
            (
                h2o,
                tuple(
                    jnp.asarray(H2O_DATA[attr])
                    for attr in Structure._get_atom_attributes()
                ),
            ),
        ],
    )
    def test_atom_attributes(
        self,
        structure: Structure,
        expected: Tuple,
    ) -> None:
        for i, attr in enumerate(Structure._get_atom_attributes()):
            if attr == "position":
                assert jnp.allclose(
                    structure.position, structure.box.shift_inside_box(expected[i])
                )
            else:
                assert jnp.allclose(getattr(structure, attr), expected[i])

    @pytest.mark.parametrize(
        "structure, expected",
        [
            (
                lj,
                (2, ("Ne",), None, jnp.float32, None),
            ),
            (
                h2o,
                (12, ("H", "O"), 11.0, _dtype.FLOATX, jnp.asarray(H2O_DATA["lattice"])),
            ),
        ],
    )
    def test_general_attributes(
        self,
        structure: Structure,
        expected: Tuple,
    ) -> None:
        assert structure.natoms == expected[0]
        assert structure.elements == expected[1]
        assert structure.r_cutoff == expected[2]
        assert structure.dtype == expected[3]
        if structure.lattice is None:
            assert structure.lattice is expected[4]
        else:
            assert jnp.allclose(structure.lattice, expected[4])
