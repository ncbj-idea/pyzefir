from typing import Any

import pandas as pd
import pytest

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network import Network
from pyzefir.model.network_elements import TransmissionFee
from tests.unit.defaults import ELECTRICITY, HEATING, default_network_constants
from tests.unit.model.test_network_elements.helpers import assert_same_exception_list


@pytest.fixture()
def network() -> Network:
    return Network(
        energy_types=[ELECTRICITY, HEATING],
        network_constants=default_network_constants,
    )


def test_transmission_fee_setup() -> None:
    """
    Test if TransmissionFee object is created correctly
    """
    transmission_fee = TransmissionFee(
        name="test_transmission_fee",
        fee=pd.Series(data=[0.0, 1.0, 0.0, 0.0]),
    )
    assert transmission_fee.name == "test_transmission_fee"
    assert transmission_fee.fee.equals(pd.Series(data=[0.0, 1.0, 0.0, 0.0]))


def test_if_all_validators_are_called(network: Network, mocker: Any) -> None:
    """
    Test if all validators are called
    """
    test_series = pd.Series(data=[0.0, 1.0, 0.0, 0.0])

    transmission_fee = TransmissionFee(
        name="test_transmission_fee",
        fee=test_series,
    )

    validate_series_mock = mocker.patch(
        "pyzefir.model.network_elements.transmission_fee.validate_series"
    )

    transmission_fee.validate(network)

    validate_series_mock.assert_called_once_with(
        name="TransmissionFee",
        series=test_series,
        length=default_network_constants.n_hours,
        allow_null=False,
        exception_list=[],
    )


def test_if_transmission_fee_correctly_raises_exception(network: Network) -> None:
    """
    Test if TransmissionFee correctly raises exception
    """
    test_series = pd.Series(data=[0.0, 1.0, 0.0, 0.0])

    transmission_fee = TransmissionFee(
        name="test_transmission_fee",
        fee=test_series,
    )

    with pytest.raises(NetworkValidatorExceptionGroup) as exception_info:
        transmission_fee.validate(network)

    assert (
        exception_info.value.message == "While adding TransmissionFee "
        "test_transmission_fee following errors occurred: "
    )

    assert_same_exception_list(
        exception_list=list(exception_info.value.exceptions),
        actual_exception_list=[
            NetworkValidatorException(
                "TransmissionFee must have 24 values",
            )
        ],
    )
