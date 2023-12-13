from typing import Any

import pandas as pd
import pytest

from pyzefir.model.network import Network
from pyzefir.model.network_elements import CapacityFactor
from tests.unit.defaults import ELECTRICITY, HEATING, default_network_constants


@pytest.fixture()
def network() -> Network:
    return Network(
        energy_types=[ELECTRICITY, HEATING],
        network_constants=default_network_constants,
    )


def test_capacity_factor_setup() -> None:
    """
    Test if CapacityFactor object is created correctly
    """
    test_series = pd.Series(data=[0.0, 1.0, 0.0, 0.0])
    capacity_factor = CapacityFactor(
        name="test_capacity_factor",
        profile=test_series,
    )
    assert capacity_factor.name == "test_capacity_factor"
    assert test_series.equals(capacity_factor.profile)


def test_if_all_validators_are_called(network: Network, mocker: Any) -> None:
    """
    Test if all validators are called
    """
    capacity_factor = CapacityFactor(
        name="test_capacity_factor",
        profile=pd.Series(data=list(range(24))),
    )
    validate_series_mock = mocker.patch(
        "pyzefir.model.network_elements.capacity_factor.validate_series",
    )
    capacity_factor.validate(network)
    validate_series_mock.assert_called_once_with(
        name="Profile",
        series=capacity_factor.profile,
        length=default_network_constants.n_hours,
        allow_null=False,
        exception_list=[],
    )
