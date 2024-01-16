import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

from typing import Tuple

import jax.numpy as jnp
import pytest
from jax import random

from pantea.descriptors.scaler import DescriptorScaler
from pantea.types import Array, default_dtype
from pantea.utils.batch import create_batch

key = random.PRNGKey(2023)
key, subkey = random.split(key, 2)


class TestStructure:
    data_1: Array = random.uniform(key, shape=(50, 3))
    data_2: Array = random.normal(subkey, shape=(100, 10), dtype=jnp.float64)
    data_3: Array = random.uniform(key, shape=(30, 5))

    @pytest.mark.parametrize(
        "data, expected",
        [
            (
                data_1,
                (3, 50, default_dtype.FLOATX),
            ),
            (
                data_2,
                (10, 100, jnp.float64),
            ),
            (
                data_3,
                (5, 30, default_dtype.FLOATX),
            ),
        ],
    )
    def test_general_attributes(
        self,
        data: Array,
        expected: Tuple,
    ) -> None:
        scaler = DescriptorScaler()
        scaler.fit(data)
        assert scaler.dimension == expected[0]
        assert scaler.stats.nsamples == expected[1]

    def fit_scaler(
        self, scaler: DescriptorScaler, data: Array, batch_size: int
    ) -> None:
        for batch in create_batch(data, batch_size=batch_size):
            scaler.fit(batch)

    def compare(self, scaler: DescriptorScaler, data: Array) -> None:
        assert jnp.allclose(data.mean(axis=0), scaler.stats.mean)
        assert jnp.allclose(data.max(axis=0), scaler.stats.maxval)
        assert jnp.allclose(data.min(axis=0), scaler.stats.minval)
        assert jnp.allclose(data.std(axis=0), scaler.stats.sigma)

    @pytest.mark.parametrize(
        "data",
        [
            data_1,
            data_2,
            data_3,
        ],
    )
    def test_array_attributes(
        self,
        data: Array,
    ) -> None:
        for scale_type, batch_size in zip(
            ("scale_center", "scale", "center"),
            (7, 10, 1),
        ):
            scaler = DescriptorScaler(scale_type=scale_type)
            self.fit_scaler(scaler, data, batch_size=batch_size)
            self.compare(scaler, data)

    @pytest.mark.parametrize(
        "data",
        [
            data_1,
            data_2,
            data_3,
        ],
    )
    def test_scaler_warnings(
        self,
        data: Array,
    ) -> None:
        scaler = DescriptorScaler()
        scaler.fit(data)

        assert scaler.max_number_of_warnings is None
        assert scaler.number_of_warnings == 0
        scaler.set_max_number_of_warnings(2)
        assert scaler.max_number_of_warnings == 2

        out_range_data: Array = jnp.ones_like(data) * 100
        scaler(out_range_data)
        assert scaler.number_of_warnings == 0
        print(out_range_data)
        scaler(out_range_data, warnings=True)
        assert scaler.number_of_warnings == 1
