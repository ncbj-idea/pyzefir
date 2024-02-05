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

from pyzefir.model.exceptions import NetworkValidatorException
from pyzefir.model.network import Network
from pyzefir.model.network_elements import EnergySourceType
from tests.unit.defaults import ELECTRICITY, HEATING, default_network_constants
from tests.unit.model.test_network_elements.helpers import assert_same_exception_list


class EnergySourceTypeTest(EnergySourceType):
    def validate(self, network: Network) -> None:
        pass


@pytest.fixture
def network() -> Network:
    network = Network(
        energy_types=[ELECTRICITY, HEATING],
        network_constants=default_network_constants,
        emission_types=["CO2", "PM10"],
    )
    return network


@pytest.fixture()
def test_energy_source_type() -> EnergySourceTypeTest:
    return EnergySourceTypeTest(
        name="test",
        life_time=5,
        capex=pd.Series([0] * default_network_constants.n_years),
        opex=pd.Series([0] * default_network_constants.n_years),
        build_time=123,
        min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        min_capacity_increase=pd.Series([np.nan] * default_network_constants.n_years),
        max_capacity_increase=pd.Series([np.nan] * default_network_constants.n_years),
    )


@pytest.mark.parametrize(
    "params, exception_list",
    [
        pytest.param(
            {},
            [],
            id="valid",
        ),
        pytest.param(
            {
                "build_time": "123",
            },
            [
                NetworkValidatorException(
                    "Energy source test has invalid build_time. Build_time must be an integer, not str"
                )
            ],
            id="invalid build_time",
        ),
        pytest.param(
            {
                "min_capacity": pd.Series([1] * default_network_constants.n_years),
                "max_capacity": pd.Series([1] * default_network_constants.n_years),
                "min_capacity_increase": pd.Series(
                    [1] * default_network_constants.n_years
                ),
                "max_capacity_increase": pd.Series(
                    [1] * default_network_constants.n_years
                ),
            },
            [
                NetworkValidatorException(
                    "Energy source type test min_capacity must have a NaN value for the base year"
                ),
                NetworkValidatorException(
                    "Energy source type test max_capacity must have a NaN value for the base year"
                ),
                NetworkValidatorException(
                    "Energy source type test min_capacity_increase must have a NaN value for the base year"
                ),
                NetworkValidatorException(
                    "Energy source type test max_capacity_increase must have a NaN value for the base year"
                ),
            ],
            id="not nan for first year",
        ),
        pytest.param(
            {
                "tags": ["abc", 12, False],
            },
            [
                NetworkValidatorException(
                    "Energy source type test has invalid tags: ['abc', 12, False]. "
                )
            ],
            id="incorrect tags values",
        ),
        pytest.param(
            {
                "tags": ("example_t_tag_1", "example_t_tag_2"),
            },
            [
                NetworkValidatorException(
                    "Energy source type test has invalid tags: ('example_t_tag_1', 'example_t_tag_2'). "
                )
            ],
            id="incorrect tags type",
        ),
    ],
)
def test_validate_base_energy_source(
    network: Network,
    test_energy_source_type: EnergySourceTypeTest,
    params: dict,
    exception_list: list[NetworkValidatorException],
) -> None:
    for key, value in params.items():
        setattr(test_energy_source_type, key, value)

    actual_exception_list: list[NetworkValidatorException] = []
    test_energy_source_type._validate_energy_source_type_base(
        network, actual_exception_list
    )
    assert_same_exception_list(actual_exception_list, exception_list)
