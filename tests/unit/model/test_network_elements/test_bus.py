from typing import Any

import pytest

from pyzefir.model.exceptions import NetworkValidatorException
from pyzefir.model.network import Network
from pyzefir.model.network_elements import Bus
from pyzefir.model.network_elements.dsr import DSR
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
    network.add_dsr(
        DSR(
            name="dsr_1",
            compensation_factor=0.1,
            balancing_period_len=10,
            penalization_minus=0.1,
            penalization_plus=0.0,
            relative_shift_limit=None,
            abs_shift_limit=None,
        )
    )
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
                    "Energy type TRANSPORT is not compliant "
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


@pytest.mark.parametrize(
    "dsr_type, expected_exception_list",
    [
        pytest.param("dsr_1", [], id="correct_dsr_type"),
        (
            pytest.param(
                1,
                [
                    NetworkValidatorException(
                        "DSR type must be type of str, not type int"
                    )
                ],
                id="incorrect_dsr_type",
            )
        ),
        pytest.param(
            "not_existing_dsr",
            [
                NetworkValidatorException(
                    "DSR type not_existing_dsr does not exist in Network DSR"
                )
            ],
            id="dsr_type_not_exist_in_network_dsr",
        ),
    ],
)
def test_validate_dsr_mapping(
    expected_exception_list: list[NetworkValidatorException] | None,
    bus: Bus,
    network: Network,
    dsr_type: Any,
) -> None:
    exception_list: list[NetworkValidatorException] = []
    bus.dsr_type = dsr_type
    bus._validate_dsr_mapping(network, exception_list)
    assert_same_exception_list(exception_list, expected_exception_list)
