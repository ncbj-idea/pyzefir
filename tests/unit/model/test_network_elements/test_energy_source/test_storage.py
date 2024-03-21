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

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network import Network
from pyzefir.model.network_elements import Bus, Fuel, Storage, StorageType
from tests.unit.defaults import (
    ELECTRICITY,
    TRANSPORT,
    default_network_constants,
    get_default_storage_type,
)
from tests.unit.model.test_network_elements.helpers import assert_same_exception_list


def test_if_all_validators_called(
    network_fixture: Network,
) -> None:
    coal = Fuel(
        name="coal",
        cost=pd.Series([0] * network_fixture.constants.n_years),
        availability=pd.Series([0] * network_fixture.constants.n_years),
        emission={},
        energy_per_unit=0.4,
    )
    network_fixture.add_fuel(coal)
    bus_a = Bus(name="bus_A", energy_type=ELECTRICITY)
    storage_type_a = get_default_storage_type(
        series_length=network_fixture.constants.n_years,
        name="default_storage_type",
        energy_type=ELECTRICITY,
    )
    network_fixture.add_bus(bus_a)
    network_fixture.add_storage_type(storage_type_a)
    storage = Storage(
        name="gen_A",
        energy_source_type="default_storage_type",
        bus="bus_A",
        unit_base_cap=25,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
    )

    storage._validate_base_energy_source = MagicMock()

    storage.validate(network_fixture)

    storage._validate_base_energy_source.assert_called_once_with(
        network=network_fixture, exception_list=[]
    )


@pytest.mark.parametrize(
    "storage, storage_type, exception_list",
    [
        (
            Storage(
                name="stor_A",
                energy_source_type="1",
                bus="bus_A",
                unit_base_cap=25,
                unit_min_capacity=pd.Series(
                    [np.nan] * default_network_constants.n_years
                ),
                unit_max_capacity=pd.Series(
                    [np.nan] * default_network_constants.n_years
                ),
                unit_min_capacity_increase=pd.Series(
                    [np.nan] * default_network_constants.n_years
                ),
                unit_max_capacity_increase=pd.Series(
                    [np.nan] * default_network_constants.n_years
                ),
            ),
            get_default_storage_type(
                series_length=default_network_constants.n_years, name="1"
            ),
            [
                NetworkValidatorException("Bus bus_A does not exist in the network"),
            ],
        ),
        (
            Storage(
                name="stor_A",
                energy_source_type="default_storage_type",
                bus="bus_A",
                unit_base_cap="sting",  # noqa
                unit_min_capacity=pd.Series(
                    [np.nan] * default_network_constants.n_years
                ),
                unit_max_capacity=pd.Series(
                    [np.nan] * default_network_constants.n_years
                ),
                unit_min_capacity_increase=pd.Series(
                    [np.nan] * default_network_constants.n_years
                ),
                unit_max_capacity_increase=pd.Series(
                    [np.nan] * default_network_constants.n_years
                ),
            ),
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                name="default_storage_type",
            ),
            [
                NetworkValidatorException(
                    "Invalid unit base capacity. Unit base capacity must be numeric, not str"
                ),
                NetworkValidatorException("Bus bus_A does not exist in the network"),
            ],
        ),
        (
            Storage(
                name="stor_A",
                energy_source_type="default_storage_type",
                bus="bus_A",
                unit_base_cap=25,
                unit_min_capacity="test_unit_cap_min",  # noqa
                unit_max_capacity="test_unit_cap_max",  # noqa
                unit_min_capacity_increase="test_unit_delta_cap_min",  # noqa
                unit_max_capacity_increase="test_unit_delta_cap_max",  # noqa
            ),
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                name="default_storage_type",
            ),
            [
                NetworkValidatorException(
                    "Unit_min_capacity must be a pandas Series, but str given"
                ),
                NetworkValidatorException(
                    "Unit_max_capacity must be a pandas Series, but str given"
                ),
                NetworkValidatorException(
                    "Unit_min_capacity_increase must be a pandas Series, but str given"
                ),
                NetworkValidatorException(
                    "Unit_max_capacity_increase must be a pandas Series, but str given"
                ),
                NetworkValidatorException("Bus bus_A does not exist in the network"),
            ],
        ),
    ],
    ids=["wrong_energy_type", "wrong_unit_base_capacity", "wrong_capacity_parameters"],
)
def test_storage(
    network_fixture: Network,
    storage: Storage,
    storage_type: StorageType,
    exception_list: list[NetworkValidatorException],
) -> None:
    network_fixture.add_storage_type(storage_type)
    with pytest.raises(NetworkValidatorExceptionGroup) as e_info:
        storage.validate(network_fixture)
    assert_same_exception_list(list(e_info.value.exceptions), exception_list)


def test_bus_energy_type_corresponds_to_storage_energy_type(
    network_fixture: Network,
) -> None:
    storage_type = get_default_storage_type(
        series_length=network_fixture.constants.n_years, name="default_storage_type"
    )
    storage = Storage(
        name="stor_A",
        energy_source_type="default_storage_type",
        bus="bus_A",
        unit_base_cap=25,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
    )
    bus = Bus(name="bus_A", energy_type=ELECTRICITY)
    exception_list = [
        NetworkValidatorException(
            "Bus bus_A energy type (TRANSPORT) is different, "
            "than energy type (ELECTRICITY) attached to this bus"
        )
    ]
    network_fixture.add_storage_type(storage_type)
    network_fixture.add_bus(bus)
    with pytest.raises(NetworkValidatorExceptionGroup) as e_info:
        bus.energy_type = TRANSPORT
        storage.validate(network_fixture)
    assert_same_exception_list(list(e_info.value.exceptions), exception_list)
