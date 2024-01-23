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
