import os
from pathlib import Path
from typing import Tuple

os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
import pytest

from jaxip.datasets import RunnerDataset
from jaxip.potentials import NeuralNetworkPotential
from jaxip.potentials.nnp.settings import (
    NeuralNetworkPotentialSettings as PotentialSettings,
)

dataset_file = Path("tests", "h2o.data")
potential_file = Path("tests", "h2o.json")


class TestNeuralNetworkPotential:
    dataset = RunnerDataset(dataset_file)
    nnp: NeuralNetworkPotential = NeuralNetworkPotential.from_file(potential_file)

    @pytest.mark.parametrize(
        "nnp, expected",
        [
            (
                nnp,
                (
                    2,
                    ("H", "O"),
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
        assert nnp.settings == PotentialSettings.from_json(potential_file)

    @pytest.mark.parametrize(
        "nnp, dataset, expected",
        [
            (
                nnp,
                dataset,
                (
                    jnp.asarray(-0.00721363),
                    jnp.asarray(
                        [
                            [-0.00343415, 0.00153666, -0.0203776],
                            [-0.11087799, -0.04327171, 0.10297609],
                            [-0.01472104, 0.0449558, 0.04925348],
                            [-0.02519826, 0.00152729, -0.01613272],
                            [0.06494806, -0.02886784, -0.01245033],
                            [-0.04200673, -0.00515676, -0.07121348],
                            [0.03067259, 0.01101292, -0.04297253],
                            [0.09775139, 0.06042628, -0.10140687],
                            [-0.02604892, 0.04006792, -0.13405925],
                            [0.01066793, -0.00437668, 0.03239093],
                            [0.05837363, 0.07088667, -0.09498966],
                            [0.04761543, -0.03721519, -0.02089787],
                        ]
                    ),
                ),
            ),
        ],
    )
    def test_outputs(
        self,
        nnp: NeuralNetworkPotential,
        dataset: RunnerDataset,
        expected: Tuple,
    ) -> None:
        nnp.fit_scaler(dataset)

        # total energy
        assert jnp.allclose(nnp(dataset[0]), expected[0])
        assert jnp.allclose(nnp.compute_forces(dataset[0]), expected[1])

    @pytest.mark.parametrize(
        "nnp, dataset",
        [
            (
                nnp,
                dataset,
            ),
        ],
    )
    def test_save_and_load(
        self,
        nnp: NeuralNetworkPotential,
        dataset: RunnerDataset,
    ) -> None:
        structure = dataset[0]
        nnp.output_dir = potential_file.parent
        nnp.save()

        settings_new = nnp.settings.copy()
        settings_new.random_seed = 4321
        nnp_new = NeuralNetworkPotential(settings=settings_new)
        nnp_new.fit_scaler(dataset)
        nnp_new.output_dir = potential_file.parent

        assert not nnp.settings == nnp_new.settings
        assert not jnp.allclose(nnp(structure), nnp_new(structure))
        nnp_new.load()
        assert jnp.allclose(nnp(structure), nnp_new(structure))
