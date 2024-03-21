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
from pyzefir.model.network_elements.emission_fee import EmissionFee
from tests.unit.defaults import (
    CO2_EMISSION,
    ELECTRICITY,
    HEATING,
    PM10_EMISSION,
    default_network_constants,
)


@pytest.fixture()
def network() -> Network:
    return Network(
        energy_types=[ELECTRICITY, HEATING],
        network_constants=default_network_constants,
        emission_types={CO2_EMISSION, PM10_EMISSION},
    )


@pytest.fixture()
def price_value() -> pd.Series:
    return pd.Series(data=[142.34, 251.32, 566.11, 102.88])


def test_emission_fee_init(price_value: pd.Series) -> None:
    emission_fee = EmissionFee(
        name="ETS1_init",
        emission_type=CO2_EMISSION,
        price=price_value,
    )
    assert isinstance(emission_fee, EmissionFee)
    assert emission_fee.name == "ETS1_init"
    assert emission_fee.emission_type == CO2_EMISSION
    assert emission_fee.price.equals(other=price_value)


def test_emission_fee_called_validators(
    network: Network, mocker: Any, price_value: pd.Series
) -> None:
    emission_fee = EmissionFee(
        name="ETS1_mock_validation",
        emission_type=CO2_EMISSION,
        price=price_value,
    )
    validate_series_mock = mocker.patch(
        "pyzefir.model.network_elements.emission_fee.validate_series"
    )
    emission_fee.validate(network)
    validate_series_mock.assert_called_once_with(
        name="EmissionFee",
        series=emission_fee.price,
        length=network.constants.n_years,
        exception_list=[],
        index_type=np.integer,
        values_type=np.floating,
        allow_null=False,
    )
    emission_fee.emission_type = "CO"
    with pytest.raises(NetworkValidatorException):
        emission_fee.validate(network)


@pytest.mark.parametrize(
    "emission_type, price",
    (
        pytest.param(CO2_EMISSION, pd.Series([0.1, 0.1, 0.1, 0.1]), id="CO2_Series"),
        pytest.param(PM10_EMISSION, pd.Series([0.2, 4.1, 2.1, 2.0]), id="PM10_Series"),
        pytest.param(PM10_EMISSION, pd.Series(np.zeros(4)), id="PM10_Series_zeroes"),
    ),
)
def test_emission_fee_validation(
    network: Network, emission_type: str, price: pd.Series
) -> None:
    emission_fee = EmissionFee(
        name="ETS_validation", emission_type=emission_type, price=price
    )
    emission_fee.validate(network=network)


@pytest.mark.parametrize(
    "emission_type, price, error_msg",
    (
        pytest.param(
            "SO2",
            pd.Series(data=[0.1, 0.1, 0.1, 0.1]),
            "Emission type: SO2 does not exist in the network",
            id="SO2 emission",
        ),
        pytest.param(
            CO2_EMISSION,
            pd.Series(data=[0.1, 0.5, 0.4, 0.12, 0.5, 0.41, 0.1]),
            "EmissionFee must have 4 values",
            id="Price lenght more than n_years",
        ),
        pytest.param(
            CO2_EMISSION,
            pd.Series(data=[0.5, 0.5, 0.5, 0.5], index=[0.1, 0.2, 0.3, 0.4]),
            "EmissionFee index type is float64 but should be integer",
            id="Price index not int",
        ),
        pytest.param(
            CO2_EMISSION,
            pd.Series(data=[15, 25, 25, 22]),
            "EmissionFee type is int64 but should be floating",
            id="Price values int not float",
        ),
        pytest.param(
            CO2_EMISSION,
            pd.Series(data=[0.2, 0.1, np.nan, 0.4]),
            "EmissionFee must not contain null values",
            id="Price contain nulls",
        ),
    ),
)
def test_emission_fee_validation_fails(
    network: Network,
    emission_type: str,
    price: pd.Series,
    error_msg: str,
) -> None:
    emission_fee = EmissionFee(name="ETS", emission_type=emission_type, price=price)
    with pytest.raises(NetworkValidatorException) as excinfo:
        emission_fee.validate(network=network)
    assert str(excinfo.value.exceptions[0]) == error_msg
