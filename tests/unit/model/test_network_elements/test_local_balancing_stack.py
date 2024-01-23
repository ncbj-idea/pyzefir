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

from unittest.mock import MagicMock

import pytest

from pyzefir.model.exceptions import NetworkValidatorException
from pyzefir.model.network import Network
from pyzefir.model.network_elements import Bus, LocalBalancingStack
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
        energy_types=[ELECTRICITY, HEATING], network_constants=default_network_constants
    )
    bus_a = Bus(name="bus_A", energy_type=ELECTRICITY)
    bus_b = Bus(name="bus_B", energy_type=ELECTRICITY)
    bus_c = Bus(name="bus_C", energy_type=HEATING)
    network.add_bus(bus_a)
    network.add_bus(bus_b)
    network.add_bus(bus_c)
    return network


def test_local_balancing_stack_should_init_correctly() -> None:
    lb_stack = LocalBalancingStack("lb_electric", buses_out={ELECTRICITY: "bus_A"})
    assert lb_stack.buses_out == {ELECTRICITY: "bus_A"}
    assert lb_stack.name == "lb_electric"


def test_if_validate_buses_out_called(network: Network) -> None:
    lb_stack = LocalBalancingStack(
        "lb_electric",
        buses_out={ELECTRICITY: "bus_A"},
        buses={ELECTRICITY: {"bus_A", "bus_B"}, HEATING: {"bus_C"}},
    )
    lb_stack._validate_buses_out = MagicMock()
    lb_stack.validate(network)
    lb_stack._validate_buses_out.assert_called_once_with(
        network=network, exception_list=[]
    )


@pytest.mark.parametrize(
    "lb_stack_params, exception_list",
    (
        pytest.param(
            {
                "name": "lb_heating",
                "buses_out": {
                    TRANSPORT: "bus_A",
                    ELECTRICITY: "bus_B",
                    HEATING: "bus_C",
                },
            },
            [
                NetworkValidatorException(
                    "Bus bus_A can not be declared as an outlet bus of local balancing "
                    f"stack lb_heating for energy {TRANSPORT}, since its energy type is {ELECTRICITY}.",
                )
            ],
            id="wrong_type",
        ),
        pytest.param(
            {
                "name": "lb_heating",
                "buses_out": {ELECTRICITY: "bus_D", HEATING: "bus_E"},
            },
            [
                NetworkValidatorException(
                    "Bus bus_D which is declared as an outlet bus of local balancing "
                    "stack lb_heating does not exist in the network.",
                ),
                NetworkValidatorException(
                    "Bus bus_E which is declared as an outlet bus of local balancing "
                    "stack lb_heating does not exist in the network.",
                ),
            ],
            id="non_existing_bus",
        ),
        pytest.param(
            {
                "name": "lb_heating",
                "buses_out": {ELECTRICITY: "bus_A", HEATING: "bus_B"},
            },
            [
                NetworkValidatorException(
                    "Bus bus_B can not be declared as an outlet bus of local balancing "
                    f"stack lb_heating for energy {HEATING}, since its energy type is "
                    f"{ELECTRICITY}.",
                )
            ],
            id="wrong_energy_type",
        ),
        pytest.param(
            {"name": "lb_bad", "buses_out": ["bus_A", "bus_B"]},
            [
                NetworkValidatorException(
                    "Outlet buses of local balancing stack lb_bad must be a dict, not <class 'list'>.",
                )
            ],
            id="wrong_buses_out_type",
        ),
        pytest.param(
            {"name": "lb_bad", "buses_out": {ELECTRICITY: 1, HEATING: "bus_A"}},
            [
                NetworkValidatorException(
                    "Outlet bus name for energy type ELECTRICITY of local balancing stack lb_bad "
                    "must be a string, not <class 'int'>.",
                )
            ],
            id="wrong_buses_out_name_type",
        ),
        pytest.param(
            {
                "name": "lb_bad",
                "buses_out": {1: "bus_A", HEATING: "bus_C", ELECTRICITY: "bus_B"},
            },
            [
                NetworkValidatorException(
                    "Energy type for outlet bus bus_A of local balancing stack lb_bad "
                    "must be a string, not <class 'int'>.",
                )
            ],
            id="wrong_buses_out_energy_type",
        ),
    ),
)
def test_validate_buses_out(
    network: Network,
    lb_stack_params: dict,
    exception_list: list[NetworkValidatorException],
) -> None:
    actual_exception_list: list[NetworkValidatorException] = []
    lb_stack = LocalBalancingStack(**lb_stack_params)
    lb_stack._validate_buses_out(network, actual_exception_list)
    assert_same_exception_list(actual_exception_list, exception_list)


def test_validate_buses_base_type() -> None:
    lb_stack = LocalBalancingStack(
        "lb_electric", buses_out={ELECTRICITY: "bus_A"}, buses="test"
    )  # noqa
    actual_exception_list: list[NetworkValidatorException] = []
    with pytest.raises(NetworkValidatorException) as e_info:
        lb_stack._validate_buses_type(actual_exception_list)
    assert str(e_info.value) == (
        "LocalBalancingStack attribute 'buses' for lb_electric must be an instance of "
        "<class 'dict'>, but it is an instance of <class 'str'> instead"
    )


@pytest.mark.parametrize(
    "lb_stack, exception_list",
    [
        (
            LocalBalancingStack(
                "lb_electric",
                buses_out={ELECTRICITY: "bus_A"},
                buses={1: {"bus_A"}, ELECTRICITY: {"bus_B"}, HEATING: {"bus_C"}},
            ),
            [
                NetworkValidatorException(
                    "All the energy types (keys of buses dict) must be a string, "
                    "but following types are found: ['int', 'str', 'str']"
                )
            ],
        ),
        (
            LocalBalancingStack(
                "lb_electric",
                buses_out={ELECTRICITY: "bus_A"},
                buses={None: {"bus_A"}, ELECTRICITY: {"bus_B", "bus_C"}},
            ),
            [
                NetworkValidatorException(
                    "All the energy types (keys of buses dict) must be a string, "
                    "but following types are found: ['NoneType', 'str']"
                )
            ],
        ),
        (
            LocalBalancingStack(
                "lb_electric",
                buses_out={ELECTRICITY: "bus_A"},
                buses={ELECTRICITY: ["bus_A", "bus_B", "bus_C"]},
            ),
            [
                NetworkValidatorException(
                    "Buses names collection type (values of buses dict) "
                    "must be a set of strings, but following types are found: ['list']"
                )
            ],
        ),
        (
            LocalBalancingStack(
                "lb_electric",
                buses_out={ELECTRICITY: "bus_A"},
                buses={ELECTRICITY: {1, "bus_B", None}},
            ),
            [
                NetworkValidatorException(
                    "Buses names collection type (values of buses dict) must contain strings only"
                )
            ],
        ),
    ],
    ids=[
        "energy_type_is_int_type",
        "energy_type_is_none_type",
        "wrong_buses_collection_type",
        "wrong_buses_name_types",
    ],
)
def test_validate_buses_subtypes(
    lb_stack: LocalBalancingStack, exception_list: list[NetworkValidatorException]
) -> None:
    actual_exception_list: list[NetworkValidatorException] = []
    lb_stack._validate_buses_type(actual_exception_list)
    assert_same_exception_list(actual_exception_list, exception_list)


@pytest.mark.parametrize(
    "lb_stack, expected_exception_list",
    [
        (
            LocalBalancingStack(
                "lb_electric",
                buses_out={ELECTRICITY: "bus_A"},
                buses={ELECTRICITY: {"bus_A", "bus_B"}, TRANSPORT: {"bus_C"}},
            ),
            [
                NetworkValidatorException(
                    "Buses energy type TRANSPORT is not defined in the Network"
                ),
                NetworkValidatorException(
                    "Energy type for bus_C in lb_electric must match with energy type for the same bus in Network"
                ),
            ],
        ),
        (
            LocalBalancingStack(
                "lb_electric",
                buses_out={ELECTRICITY: "bus_A"},
                buses={TRANSPORT: {"bus_A"}},
            ),
            [
                NetworkValidatorException(
                    "Buses energy type TRANSPORT is not defined in the Network"
                ),
                NetworkValidatorException(
                    "Energy type for bus_A in lb_electric must match with energy type for the same bus in Network"
                ),
            ],
        ),
        (
            LocalBalancingStack(
                "lb_electric",
                buses_out={ELECTRICITY: "bus_A"},
                buses={ELECTRICITY: {"bus_TEST"}},
            ),
            [
                NetworkValidatorException(
                    "Bus name 'bus_TEST' must exist in the Network"
                )
            ],
        ),
        (
            LocalBalancingStack(
                "lb_electric",
                buses_out={ELECTRICITY: "bus_A"},
                buses={ELECTRICITY: {"bus_A", "bus_C"}, HEATING: {"bus_B"}},
            ),
            [
                NetworkValidatorException(
                    "Energy type for bus_C in lb_electric must match with energy type for the same bus in Network"
                ),
                NetworkValidatorException(
                    "Energy type for bus_B in lb_electric must match with energy type for the same bus in Network"
                ),
            ],
        ),
    ],
    ids=[
        "wrong_buses_energy_type_1",
        "wrong_buses_energy_type_2",
        "bus_name_not_in_network",
        "bus_energy_type_doesnt_match_with_energy_type_for_the_same_bus_in_network",
    ],
)
def test_validate_buses(
    network: Network,
    lb_stack: LocalBalancingStack,
    expected_exception_list: list[NetworkValidatorException],
) -> None:
    actual_exception_list: list[NetworkValidatorException] = []
    lb_stack._validate_buses(network, actual_exception_list)
    assert_same_exception_list(actual_exception_list, expected_exception_list)
