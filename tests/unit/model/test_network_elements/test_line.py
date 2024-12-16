from typing import Any

import pandas as pd
import pytest

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network import Network
from pyzefir.model.network_elements import Bus, Line, TransmissionFee
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
def line() -> Line:
    return Line(
        name="line_1",
        energy_type=ELECTRICITY,
        fr="bus_A",
        to="bus_B",
        transmission_loss=0.5,
        max_capacity=1.5,
    )


def test_validate_energy_type(line: Line, network: Network) -> None:
    exception_list: list[NetworkValidatorException] = []
    line.energy_type = ELECTRICITY
    line._validate_energy_type(network, exception_list)

    assert len(exception_list) == 0

    line.energy_type = "non_existent"
    line._validate_energy_type(network, exception_list)

    assert len(exception_list) == 1
    assert str(exception_list[0]) == (
        "Energy type of line non_existent not found in the Network energy types: ['ELECTRICITY', 'HEATING']"
    )


@pytest.mark.parametrize(
    "fr, to, energy_type, expected_exception_list",
    (
        ("bus_A", "bus_B", ELECTRICITY, None),
        (
            "bus_A",
            "bus_C",
            ELECTRICITY,
            [
                NetworkValidatorException(
                    "Cannot set end of the line to bus bus_C. "
                    "Bus bus_C energy type is HEATING, which is different from "
                    "the line energy type: ELECTRICITY."
                ),
                NetworkValidatorException(
                    "Cannot add a line between buses bus_A and bus_C with "
                    "different energy types ELECTRICITY != HEATING"
                ),
            ],
        ),
        (
            "bus_404",
            "bus_501",
            TRANSPORT,
            [
                NetworkValidatorException(
                    "Cannot set the beginning of the line to bus bus_404. "
                    "Bus bus_404 does not exist in the network"
                ),
                NetworkValidatorException(
                    "Cannot set the end of the line to bus bus_501. "
                    "Bus bus_501 does not exist in the network"
                ),
            ],
        ),
        (
            "bus_404",
            "bus_B",
            ELECTRICITY,
            [
                NetworkValidatorException(
                    "Cannot set the beginning of the line to bus bus_404. "
                    "Bus bus_404 does not exist in the network"
                )
            ],
        ),
    ),
)
def test_validate_line_connections(
    fr: str,
    to: str,
    energy_type: str,
    expected_exception_list: list[NetworkValidatorException] | None,
    line: Line,
    network: Network,
) -> None:
    line = Line(
        name=line.name,
        energy_type=energy_type,
        fr=fr,
        to=to,
        max_capacity=line.max_capacity,
        transmission_loss=line.transmission_loss,
    )
    exception_list: list[NetworkValidatorException] = []

    line._validate_line_connections(network, exception_list)

    if expected_exception_list is None:
        assert len(exception_list) == 0
    else:
        assert_same_exception_list(exception_list, expected_exception_list)


@pytest.mark.parametrize(
    "transmission_loss, expected_exception_list",
    (
        (1.0, None),
        (
            "1",
            [
                NetworkValidatorException(
                    "Transmission loss must be of type float, but is <class 'str'> instead"
                )
            ],
        ),
        (
            -1,
            [
                NetworkValidatorException(
                    "The value of the transmission_loss is inconsistent with th expected bounds of the "
                    "interval: 0 <= -1 <= 1",
                )
            ],
        ),
        (
            2,
            [
                NetworkValidatorException(
                    "The value of the transmission_loss is inconsistent with th expected bounds of the "
                    "interval: 0 <= 2 <= 1",
                )
            ],
        ),
        (
            None,
            [
                NetworkValidatorException(
                    "Transmission loss must be of type float, but is <class 'NoneType'> instead"
                )
            ],
        ),
    ),
)
def test_validate_transmission_loss(
    transmission_loss: Any,
    expected_exception_list: list[NetworkValidatorException] | None,
    line: Line,
) -> None:
    exception_list: list[NetworkValidatorException] = []
    line.transmission_loss = transmission_loss
    line._validate_transmission_loss(exception_list)

    if expected_exception_list is None:
        assert len(exception_list) == 0
    else:
        assert_same_exception_list(exception_list, expected_exception_list)


def test_validate_max_capacity(line: Line) -> None:
    exception_list: list[NetworkValidatorException] = []
    line.max_capacity = 5.6
    line._validate_max_capacity(exception_list)

    assert len(exception_list) == 0

    line.max_capacity = "test"
    line._validate_max_capacity(exception_list)

    assert len(exception_list) == 1
    assert (
        str(exception_list[0])
        == "Max capacity must be of type float, but is <class 'str'> instead"
    )


def test_validate_transmission_fee(network: Network) -> None:
    network.add_transmission_fee(
        TransmissionFee(
            name="fee_1", fee=pd.Series(data=[1.0] * default_network_constants.n_hours)
        )
    )

    line_correct = Line(
        name="line_1",
        energy_type=ELECTRICITY,
        fr="bus_A",
        to="bus_B",
        transmission_loss=0.5,
        max_capacity=1.5,
        transmission_fee="fee_1",
    )

    line_incorrect = Line(
        name="line_1",
        energy_type=ELECTRICITY,
        fr="bus_A",
        to="bus_B",
        transmission_loss=0.5,
        max_capacity=1.5,
        transmission_fee="fee_2",
    )

    line_correct.validate(network)

    with pytest.raises(NetworkValidatorExceptionGroup) as exc_info:
        line_incorrect.validate(network)

    assert_same_exception_list(
        actual_exception_list=list(exc_info.value.exceptions),
        exception_list=[
            NetworkValidatorException(
                "Cannot set a transmission fee for the line. "
                "Transmission fee fee_2 does not exist in the network"
            )
        ],
    )
