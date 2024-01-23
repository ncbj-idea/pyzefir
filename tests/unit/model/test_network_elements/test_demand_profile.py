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

import pandas as pd
import pytest

from pyzefir.model.exceptions import NetworkValidatorException
from pyzefir.model.network import Network
from pyzefir.model.network_elements import DemandProfile
from tests.unit.defaults import (
    ELECTRICITY,
    HEATING,
    default_network_constants,
    default_yearly_demand,
)
from tests.unit.model.test_network_elements.helpers import assert_same_exception_list


@pytest.fixture()
def network() -> Network:
    return Network(
        energy_types=[ELECTRICITY, HEATING],
        network_constants=default_network_constants,
    )


def test_demand_profile_setup(network: Network) -> None:
    """
    Test if DemandProfile object is created correctly
    """
    demand_profile = DemandProfile(
        name="test_demand_profile",
        normalized_profile=default_yearly_demand,
    )
    assert demand_profile.name == "test_demand_profile"
    assert demand_profile.normalized_profile == default_yearly_demand


def test_if_all_validators_are_called(network: Network) -> None:
    """
    Test if all validators are called
    """
    demand_profile = DemandProfile(
        name="test_demand_profile",
        normalized_profile=default_yearly_demand,
    )

    demand_profile._validate_normalized_profile = MagicMock()

    demand_profile.validate(network)

    demand_profile._validate_normalized_profile.assert_called_once_with(
        network=network, exception_list=[]
    )


@pytest.mark.parametrize(
    "normalized_profile, exception_list",
    [
        pytest.param(
            {
                ELECTRICITY: pd.Series(data=[0.0, 1.0, 0.0, 0.0]),
                HEATING: pd.Series(data=[0.0, 0.4, 0.5, 0.10]),
            },
            [],
            id="correct profile",
        ),
        pytest.param(
            {
                ELECTRICITY: pd.Series(data=[0.0, 1.0, 0.0, 0.0]),
                HEATING: pd.Series(data=[0.0, 0.4, 0.5, 0.1]),
                "PV": pd.Series(data=[0.0, 1.0, 0.0, 0.0]),
            },
            [],
            id="more energy types than in network",
        ),
        pytest.param(
            {
                ELECTRICITY: pd.Series(data=[0.0, 1.1, 0.0, -0.1]),
                HEATING: pd.Series(data=[0.0, 1.0, 0.5, 0.10]),
            },
            [
                NetworkValidatorException(
                    "Energy type ELECTRICITY in test_demand_profile is not normalized"
                ),
                NetworkValidatorException(
                    "Energy type HEATING in test_demand_profile is not normalized"
                ),
            ],
            id="not normalized and does not sum to 1",
        ),
        pytest.param(
            {
                ELECTRICITY: pd.Series(data=[0.0, 0.4, 0.5, 0.1]),
                HEATING: pd.Series(data=[0.0, 1.0, 0.0]),
            },
            [
                NetworkValidatorException(
                    "Normalized profile in test_demand_profile has different "
                    "length for different energy types"
                )
            ],
            id="different length",
        ),
        pytest.param(
            {
                ELECTRICITY: pd.Series(data=[0.0, 0.4, 0.1, 0.1]),
                HEATING: pd.Series(data=[0.0, 1.1, -0.1]),
            },
            [
                NetworkValidatorException(
                    "Normalized profile in test_demand_profile has different "
                    "length for different energy types"
                ),
                NetworkValidatorException(
                    "Energy type ELECTRICITY in test_demand_profile is not normalized"
                ),
                NetworkValidatorException(
                    "Energy type HEATING in test_demand_profile is not normalized"
                ),
            ],
            id="different length and not normalized",
        ),
    ],
)
def test_validate_normalized_profile(
    network: Network,
    normalized_profile: dict[str, pd.Series],
    exception_list: list[NetworkValidatorException],
) -> None:
    """
    Test if _validate_normalized_profile method works correctly
    """
    demand_profile = DemandProfile(
        name="test_demand_profile",
        normalized_profile=normalized_profile,
    )

    test_exception_list: list[NetworkValidatorException] = []

    demand_profile._validate_normalized_profile(
        network=network, exception_list=test_exception_list
    )

    assert_same_exception_list(test_exception_list, exception_list)
