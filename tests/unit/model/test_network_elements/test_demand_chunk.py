# PyZefir
# Copyright (C) 2023-2024 Narodowe Centrum Badań Jądrowych
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network import Network
from pyzefir.model.network_elements import DemandChunk, Generator, Storage
from tests.unit.defaults import (
    CO2_EMISSION,
    ELECTRICITY,
    HEATING,
    PM10_EMISSION,
    default_network_constants,
)
from tests.unit.model.test_network_elements.helpers import assert_same_exception_list


@pytest.fixture()
def network() -> Network:
    network = Network(
        emission_types=[CO2_EMISSION, PM10_EMISSION],
        energy_types=[ELECTRICITY, HEATING],
        network_constants=default_network_constants,
    )

    for gen_name, tags in [
        ("gen_A", ["tag_A", "tag_B"]),
        ("gen_B", ["tag_B", "tag_C"]),
    ]:
        network.generators[gen_name] = Generator(
            name=gen_name,
            energy_source_type="test_energy_source_type",
            bus={"bus_A", "bus_B"},
            unit_base_cap=25,
            unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
            unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
            unit_min_capacity_increase=pd.Series(
                [np.nan] * default_network_constants.n_years
            ),
            unit_max_capacity_increase=pd.Series(
                [np.nan] * default_network_constants.n_years
            ),
            tags=tags,
        )

    for stor_name, tags in [
        ("stor_A", ["tag_E", "tag_F"]),
        ("stor_B", ["tag_F", "tag_G"]),
    ]:
        network.storages[stor_name] = Storage(
            name=stor_name,
            energy_source_type="test_energy_source_type",
            bus="bus_A",
            unit_base_cap=25,
            unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
            unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
            unit_min_capacity_increase=pd.Series(
                [np.nan] * default_network_constants.n_years
            ),
            unit_max_capacity_increase=pd.Series(
                [np.nan] * default_network_constants.n_years
            ),
            tags=tags,
        )

    return network


@pytest.mark.parametrize(
    "params, exception_list",
    [
        pytest.param(
            {
                "name": "test_chunk",
                "energy_type": ELECTRICITY,
                "periods": np.array([[1, 2], [3, 4]]),
                "demand": np.array([[10, 2, 2, 3], [3, 4, 5, 6]]),
                "tag": "tag_A",
            },
            [],
            id="correct_demand_chunk",
        ),
        pytest.param(
            {
                "name": "test_chunk",
                "energy_type": ELECTRICITY,
                "periods": np.array([[1, 2], [3, 4]]),
                "demand": np.array([[10, 2, 2, 3], [3, 4, 5, 6]]),
                "tag": "tag_D",
            },
            [NetworkValidatorException("Tag tag_D is not defined in the network")],
            id="incorrect_tag",
        ),
        pytest.param(
            {
                "name": "test_chunk",
                "energy_type": ELECTRICITY,
                "periods": np.array([[1], [3], [5]]),
                "demand": np.array([[10, 2], [3, 4]]),
                "tag": "tag_E",
            },
            [
                NetworkValidatorException(
                    "Length of periods (3) and demand (2) should be the same"
                ),
                NetworkValidatorException("Periods should have 2 columns, not 1"),
                NetworkValidatorException("Demand should have 4 columns, not 2"),
            ],
            id="incorrect_shape_of_demand_and_periods",
        ),
        pytest.param(
            {
                "name": "test_chunk",
                "energy_type": ELECTRICITY,
                "periods": np.array([[2, 2], [3, 4]]),
                "demand": np.array([[10, 2, 2, 3], [3, 4, 5, 6]]),
                "tag": "tag_A",
            },
            [
                NetworkValidatorException(
                    "Periods should be in the format (start, end), not (end, start)"
                )
            ],
            id="incorrect_values_of_periods",
        ),
        pytest.param(
            {
                "name": "test_chunk",
                "energy_type": ELECTRICITY,
                "periods": np.array([[2.1, 2.5], [3, 4]]),
                "demand": np.array([["10", "2", "2", "asd"], ["3", "4", "5", "6"]]),
                "tag": "tag_A",
            },
            [
                NetworkValidatorException("Periods should be type of int, not float64"),
                NetworkValidatorException("Demand should be type of float, not <U3"),
            ],
            id="incorrect_types",
        ),
        pytest.param(
            {
                "name": "test_chunk",
                "energy_type": 123,
                "periods": np.array([[1, 2], [3, 4]]),
                "demand": np.array([[10, 2, 2, 3], [3, 4, 5, 6]]),
                "tag": 123,
            },
            [
                NetworkValidatorException("Tag 123 should be type of str, not int"),
                NetworkValidatorException(
                    "Energy type 123 should be type of str, not int"
                ),
            ],
            id="bad_types",
        ),
        pytest.param(
            {
                "name": "test_chunk",
                "energy_type": "123",
                "periods": np.array([[1, 2], [3, 4]]),
                "demand": np.array([[10, 2, 2, 3], [3, 4, 5, 6]]),
                "tag": "123",
            },
            [
                NetworkValidatorException("Tag 123 is not defined in the network"),
                NetworkValidatorException(
                    "Energy type 123 is not defined in the network"
                ),
            ],
            id="not_defined_tag_and_energy_type",
        ),
    ],
)
def test_demand_chunk_validation(
    params: dict, exception_list: list[NetworkValidatorException], network: Network
) -> None:
    demand_chunk = DemandChunk(**params)
    try:
        demand_chunk.validate(network=network)
    except NetworkValidatorExceptionGroup as e:
        assert_same_exception_list(list(e.exceptions), exception_list)
