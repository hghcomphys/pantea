import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

from typing import Tuple

import jax.numpy as jnp
import pytest
from jax import random

from pantea.descriptors.scaler import DescriptorScaler, ScalerParams
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
    def test_params_sizes(
        self,
        data: Array,
        expected: Tuple,
    ) -> None:
        scaler = DescriptorScaler.from_type("scale_center")
        params = scaler.fit(data)
        assert int(params.dimension) == expected[0]
        assert int(params.nsamples) == expected[1]

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
            scaler = DescriptorScaler.from_type(scale_type)
            params = self.fit_scaler(scaler, data, batch_size=batch_size)
            self.check_values(params, data)

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
        scaler = DescriptorScaler.from_type("scale_center")
        params = scaler.fit(data)
        warnings = scaler.initialize_warnings()
        warnings = scaler.check_warnings(params, data, warnings)
        assert warnings.max_number_of_warnings < 0
        assert warnings.number_of_warnings == 0

        warnings = scaler.initialize_warnings(max_number_of_warnings=2)
        assert warnings.max_number_of_warnings == 2

        outlier_data = jnp.ones_like(data) * 100
        warnings = scaler.check_warnings(params, outlier_data, warnings)
        assert warnings.number_of_warnings == 1

    def fit_scaler(
        self, scaler: DescriptorScaler, data: Array, batch_size: int
    ) -> ScalerParams:
        batches = create_batch(data, batch_size)
        params = scaler.fit(data=next(batches))
        for batch in batches:
            params = scaler.partial_fit(params, batch)
        return params

    def check_values(self, params: ScalerParams, data: Array) -> None:
        assert jnp.allclose(data.mean(axis=0), params.mean)
        assert jnp.allclose(data.max(axis=0), params.maxval)
        assert jnp.allclose(data.min(axis=0), params.minval)
        assert jnp.allclose(data.std(axis=0), params.sigma)
