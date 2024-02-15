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

import copy

import pandas as pd
import pytest

from pyzefir.model.exceptions import NetworkValidatorException
from pyzefir.model.network import Network
from pyzefir.model.network_elements import Bus, Fuel
from tests.unit.defaults import ELECTRICITY, HEATING, default_network_constants
from tests.unit.model.test_network_elements.helpers import assert_same_exception_list
from tests.unit.optimization.linopy.names import CO2, PM10


@pytest.fixture()
def network() -> Network:
    network = Network(
        energy_types=[ELECTRICITY, HEATING],
        network_constants=default_network_constants,
        emission_types=[PM10, CO2],
    )
    network.add_bus(Bus(name="bus_A", energy_type=ELECTRICITY))
    network.add_bus(Bus(name="bus_B", energy_type=ELECTRICITY))
    network.add_bus(Bus(name="bus_C", energy_type=HEATING))

    return network


@pytest.fixture()
def fuel() -> Fuel:
    return Fuel(
        name="fuel_1",
        emission=dict(),
        availability=pd.Series([0] * default_network_constants.n_years),
        cost=pd.Series([0] * default_network_constants.n_years),
        energy_per_unit=5.0,
    )


def test_validate_emission(fuel: Fuel, network: Network) -> None:
    exception_list: list[NetworkValidatorException] = []
    fuel.emission = {PM10: 5, CO2: 0.5}
    fuel._validate_emission(network, exception_list)

    assert len(exception_list) == 0

    fuel.emission = {"non_existent1": 0.5, "404": 10}
    fuel._validate_emission(network, exception_list)

    assert len(exception_list) == 2
    assert str(exception_list[0]) == "Emission type non_existent1 not found in network"
    assert str(exception_list[1]) == "Emission type 404 not found in network"


@pytest.mark.parametrize(
    "attr_name, attr_value, expected_exception_list",
    (
        (
            "availability",
            pd.Series(),
            [
                NetworkValidatorException("Availability must have 4 values"),
            ],
        ),
        (
            "cost",
            pd.Series([0] * 999),
            [
                NetworkValidatorException("Cost must have 4 values"),
            ],
        ),
        (
            "energy_per_unit",
            "string",
            [
                NetworkValidatorException("Energy per unit must be of float type"),
            ],
        ),
        (
            "emission",
            {5: "abc"},
            [
                NetworkValidatorException(
                    "Emission type in Emission mapping must be of <class 'str'> type"
                ),
                NetworkValidatorException(
                    "Emission per unit in Emission mapping must be of float | int type"
                ),
            ],
        ),
    ),
)
def test_validate_attributes(
    attr_name: str,
    attr_value: str,
    expected_exception_list: list[NetworkValidatorException] | None,
    fuel: Fuel,
    network: Network,
) -> None:
    fuel = copy.deepcopy(fuel)
    setattr(fuel, attr_name, attr_value)
    exception_list: list[NetworkValidatorException] = []

    fuel._validate_attributes(network, exception_list)

    if expected_exception_list is None:
        assert len(exception_list) == 0
    else:
        assert_same_exception_list(exception_list, expected_exception_list)
