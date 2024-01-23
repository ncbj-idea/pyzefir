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

import pytest

from pyzefir.model.exceptions import NetworkValidatorException
from pyzefir.model.network import Network
from pyzefir.model.network_elements import Bus
from tests.unit.defaults import (
    ELECTRICITY,
    HEATING,
    TRANSPORT,
    default_network_constants,
)
from tests.unit.model.test_network_elements.helpers import assert_same_exception_list


@pytest.fixture()
def network() -> Network:
    network = Network(
        energy_types=[ELECTRICITY, HEATING],
        network_constants=default_network_constants,
    )
    network.add_bus(Bus(name="bus_A", energy_type=ELECTRICITY))
    network.add_bus(Bus(name="bus_B", energy_type=ELECTRICITY))
    network.add_bus(Bus(name="bus_C", energy_type=HEATING))

    return network


@pytest.fixture()
def bus() -> Bus:
    return Bus(
        name="bus_1",
        energy_type=ELECTRICITY,
    )


def test_bus_init(bus: Bus) -> None:
    assert isinstance(bus.name, str)
    assert isinstance(bus.generators, set) and len(bus.generators) == 0
    assert isinstance(bus.storages, set) and len(bus.storages) == 0
    assert isinstance(bus.lines_in, set) and len(bus.lines_in) == 0
    assert isinstance(bus.lines_out, set) and len(bus.lines_out) == 0


@pytest.mark.parametrize(
    "energy_type, expected_exception_list",
    (
        (ELECTRICITY, None),
        (HEATING, None),
        (
            None,
            [
                NetworkValidatorException(
                    "Energy Type must be a string, but given <class 'NoneType'> instead"
                )
            ],
        ),
        (
            TRANSPORT,
            [
                NetworkValidatorException(
                    "Bus bus_1 has energy type TRANSPORT which is not compliant "
                    "with the network energy types: ['ELECTRICITY', 'HEATING']"
                )
            ],
        ),
    ),
)
def test_validate_energy_type(
    energy_type: Any,
    expected_exception_list: list[NetworkValidatorException] | None,
    bus: Bus,
    network: Network,
) -> None:
    exception_list: list[NetworkValidatorException] = []
    bus.energy_type = energy_type
    bus._validate_energy_type(network, exception_list)

    if expected_exception_list is None:
        assert len(exception_list) == 0
    else:
        assert_same_exception_list(exception_list, expected_exception_list)
