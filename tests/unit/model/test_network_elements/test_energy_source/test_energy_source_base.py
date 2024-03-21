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

from typing import Any

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.exceptions import NetworkValidatorException
from pyzefir.model.network import Network
from pyzefir.model.network_elements import EnergySource
from tests.unit.defaults import ELECTRICITY, HEATING, default_network_constants
from tests.unit.model.test_network_elements.helpers import assert_same_exception_list


class EnergySourceTest(EnergySource):
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
def test_energy_source() -> EnergySource:
    return EnergySourceTest(
        name="test",
        energy_source_type="test",
        unit_base_cap=15,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
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
                "energy_source_type": 123,
            },
            [
                NetworkValidatorException(
                    "Invalid energy source type. Energy source type must be a string, not int"
                )
            ],
            id="invalid energy_source_type",
        ),
        pytest.param(
            {
                "unit_base_cap": "123",
            },
            [
                NetworkValidatorException(
                    "Invalid unit base capacity. Unit base capacity must be numeric, not str"
                )
            ],
            id="invalid unit_base_cap",
        ),
        pytest.param(
            {
                "unit_min_capacity": pd.Series([1] * default_network_constants.n_years),
                "unit_max_capacity": pd.Series([1] * default_network_constants.n_years),
                "unit_min_capacity_increase": pd.Series(
                    [1] * default_network_constants.n_years
                ),
                "unit_max_capacity_increase": pd.Series(
                    [1] * default_network_constants.n_years
                ),
            },
            [
                NetworkValidatorException(
                    "Unit_min_capacity must have a NaN value for the base year"
                ),
                NetworkValidatorException(
                    "Unit_max_capacity must have a NaN value for the base year"
                ),
                NetworkValidatorException(
                    "Unit_min_capacity_increase must have a NaN value for the base year"
                ),
                NetworkValidatorException(
                    "Unit_max_capacity_increase must have a NaN value for the base year"
                ),
            ],
            id="not nan for first year",
        ),
        pytest.param(
            {
                "tags": [0, None, "abc"],
            },
            [
                NetworkValidatorException("Invalid tags: [0, None, 'abc']. "),
            ],
            id="incorrect tags",
        ),
    ],
)
def test_validate_base_energy_source(
    network: Network,
    params: dict,
    test_energy_source: EnergySourceTest,
    exception_list: list[NetworkValidatorException],
) -> None:
    for key, value in params.items():
        setattr(test_energy_source, key, value)
    actual_exception_list: list[NetworkValidatorException] = []
    test_energy_source._validate_base_energy_source(network, actual_exception_list)
    assert_same_exception_list(actual_exception_list, exception_list)


def test_correct_series_validation(network: Network, mocker: Any) -> None:
    test_energy_source = EnergySourceTest(
        name="test",
        energy_source_type="test",
        unit_base_cap=123,
        unit_min_capacity=pd.Series([np.nan, 2, 3]),
        unit_max_capacity=pd.Series([np.nan, 2, 3]),
        unit_min_capacity_increase=pd.Series([np.nan, 2, 3]),
        unit_max_capacity_increase=pd.Series([np.nan, 2, 3]),
    )

    validate_series_mock = mocker.patch(
        "pyzefir.model.network_elements.energy_sources.energy_source_base.validate_series",
        return_value=True,
    )

    test_energy_source._validate_base_energy_source(network=network, exception_list=[])

    validate_series_mock.assert_has_calls(
        [
            mocker.call(
                series=getattr(test_energy_source, attr),
                name=f"{attr.capitalize()}",
                exception_list=[],
                length=default_network_constants.n_years,
                is_numeric=True,
            )
            for attr in [
                "unit_min_capacity",
                "unit_max_capacity",
                "unit_min_capacity_increase",
                "unit_max_capacity_increase",
            ]
        ]
    )


@pytest.mark.parametrize(
    "min_device_nom_power, max_device_nom_power, expected_exception_list",
    [
        (
            "test_min_dev_nom_power",
            "test_max_dev_nom_power",
            [
                NetworkValidatorException(
                    "Invalid min_device_nom_power. "
                    "min_device_nom_power must be an instance of "
                    "one of the types: float, int or None, not str"
                ),
                NetworkValidatorException(
                    "Invalid max_device_nom_power. "
                    "max_device_nom_power must be an instance of "
                    "one of the types: float, int or None, not str"
                ),
            ],
        ),
        (
            -1.0,
            0,
            [
                NetworkValidatorException(
                    "Min_device_nom_power has invalid value. "
                    "It must be greater or equal to zero, but it is: -1.0"
                ),
            ],
        ),
        (
            0.0,
            2,
            [],
        ),
    ],
    ids=[
        "invalid_both_device_nom_power_type",
        "invalid_min_device_nom_power_value",
        "both_device_nom_power_valid",
    ],
)
def test_device_nominal_power(
    network: Network,
    min_device_nom_power: Any,
    max_device_nom_power: Any,
    expected_exception_list: list[NetworkValidatorException],
) -> None:
    actual_exception_list: list[NetworkValidatorException] = []
    energy_source_test = EnergySourceTest(
        name="test",
        energy_source_type="test",
        unit_base_cap=123,
        min_device_nom_power=min_device_nom_power,
        max_device_nom_power=max_device_nom_power,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
    )
    energy_source_test._validate_device_nominal_power(
        "min_device_nom_power", actual_exception_list
    )
    energy_source_test._validate_device_nominal_power(
        "max_device_nom_power", actual_exception_list
    )
    assert_same_exception_list(actual_exception_list, expected_exception_list)
