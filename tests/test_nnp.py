import os
from pathlib import Path
from typing import Tuple

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
import pytest

from jaxip.datasets import RunnerStructureDataset
from jaxip.potentials import NeuralNetworkPotential
from jaxip.potentials import NeuralNetworkPotentialSettings as Settings

dataset_file = Path("tests", "h2o.data")
potential_file = Path("tests", "h2o.json")


class TestNeuralNetworkPotential:

    # TODO: add more tests
    dataset = RunnerStructureDataset(dataset_file)
    nnp = NeuralNetworkPotential(Settings.from_json(potential_file))  # type: ignore

    @pytest.mark.parametrize(
        "nnp, expected",
        [
            (
                nnp,
                (
                    2,
                    ["H", "O"],
                ),
            ),
        ],
    )
    def test_settings(
        self,
        nnp: NeuralNetworkPotential,
        expected: Tuple,
    ) -> None:
        assert nnp.num_elements == expected[0]
        assert nnp.elements == expected[1]
        assert nnp.settings == Settings.from_json(potential_file)

    @pytest.mark.parametrize(
        "nnp, dataset, expected",
        [
            (
                nnp,
                dataset,
                (
                    jnp.asarray(-0.00721363),
                    {
                        "H": jnp.asarray(
                            [
                                [-0.11087801, -0.04327172, 0.1029761],
                                [-0.01472103, 0.04495581, 0.04925349],
                                [0.06494806, -0.02886785, -0.01245032],
                                [-0.0420067, -0.00515675, -0.07121348],
                                [0.09775142, 0.06042631, -0.10140688],
                                [-0.02604893, 0.04006793, -0.13405928],
                                [0.05837363, 0.07088665, -0.09498966],
                                [0.04761543, -0.03721521, -0.02089789],
                            ]
                        ),
                        "O": jnp.asarray(
                            [
                                [-0.00343414, 0.00153665, -0.02037759],
                                [-0.02519827, 0.00152728, -0.01613272],
                                [0.0306726, 0.01101293, -0.04297253],
                                [0.01066793, -0.00437668, 0.03239094],
                            ]
                        ),
                    },
                ),
            ),
        ],
    )
    def test_outputs(
        self,
        nnp: NeuralNetworkPotential,
        dataset: RunnerStructureDataset,
        expected: Tuple,
    ) -> None:
        nnp.fit_scaler(dataset)

        # total energy
        assert jnp.allclose(nnp(dataset[0]), expected[0])

        # force components for each element
        force = nnp.compute_force(dataset[0])
        for element in nnp.elements:
            assert jnp.allclose(force[element], expected[1][element])
