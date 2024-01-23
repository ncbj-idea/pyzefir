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

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network import Network
from pyzefir.model.network_elements.energy_source_types.storage_type import StorageType
from pyzefir.model.network_elements.energy_sources.storage import Storage
from tests.unit.defaults import default_network_constants, get_default_storage_type
from tests.unit.model.test_network_elements.helpers import assert_same_exception_list


@pytest.mark.parametrize(
    "storage_type, exception_list",
    [
        (
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                generation_efficiency="string",
                name="default_storage_type",
            ),
            [
                NetworkValidatorException(
                    "StorageType attribute 'generation_efficiency' for default_storage_type must be an instance "
                    "of float | int, but it is an instance of <class 'str'> instead"
                )
            ],
        ),
        (
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                load_efficiency="string",
                name="default_storage_type",
            ),
            [
                NetworkValidatorException(
                    "StorageType attribute 'load_efficiency' for default_storage_type must be an instance "
                    "of float | int, but it is an instance of <class 'str'> instead"
                )
            ],
        ),
        (
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                cycle_length="string",
                name="default_storage_type",
            ),
            [
                NetworkValidatorException(
                    "StorageType attribute 'cycle_length' for default_storage_type must be an instance "
                    "of <class 'int'>, but it is an instance of <class 'str'> instead"
                )
            ],
        ),
        (
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                power_to_capacity="string",
                name="default_storage_type",
            ),
            [
                NetworkValidatorException(
                    "StorageType attribute 'power_to_capacity' for default_storage_type must be an instance "
                    "of float | int, but it is an instance of <class 'str'> instead"
                )
            ],
        ),
        (
            get_default_storage_type(
                series_length=default_network_constants.n_years,
                energy_type=1,
                name="default_storage_type",
            ),
            [
                NetworkValidatorException(
                    "StorageType attribute 'energy_type' for default_storage_type must be an instance "
                    "of <class 'str'>, but it is an instance of <class 'int'> instead"
                ),
                NetworkValidatorException(
                    "StorageType default_storage_type has energy type 1 "
                    "which is not compliant with the network energy types: ['ELECTRICITY', 'HEATING']"
                ),
            ],
        ),
    ],
    ids=[
        "generation_efficiency_wrong_type",
        "load_efficiency_wrong_type",
        "cycle_length_wrong_type",
        "power_to_capacity_wrong_type",
        "energy_type_wrong_type",
    ],
)
def test_attribute_validators(
    storage_type: StorageType,
    exception_list: list[NetworkValidatorException],
    network_fixture: Network,
) -> None:
    with pytest.raises(NetworkValidatorExceptionGroup) as e_info:
        storage_type.validate(network_fixture)
    assert_same_exception_list(list(e_info.value.exceptions), exception_list)


@pytest.mark.parametrize(
    "storage_type, storage, expected_exception_list",
    [
        (
            None,
            Storage(
                name="storage_a",
                bus="bus_A",
                energy_source_type="test_storage_type",
                unit_base_cap=15,
                unit_min_capacity=pd.Series(
                    np.empty(default_network_constants.n_years).fill(np.nan)
                ),
                unit_max_capacity=pd.Series(
                    np.empty(default_network_constants.n_years).fill(np.nan)
                ),
                unit_min_capacity_increase=pd.Series(
                    np.empty(default_network_constants.n_years).fill(np.nan)
                ),
                unit_max_capacity_increase=pd.Series(
                    np.empty(default_network_constants.n_years).fill(np.nan)
                ),
            ),
            [
                NetworkValidatorException(
                    "Storage type test_storage_type not found in the network"
                ),
            ],
        ),
        (
            "string",
            Storage(
                name="storage_a",
                bus="bus_A",
                energy_source_type="test_storage_type",
                unit_base_cap=15,
                unit_min_capacity=pd.Series(
                    np.empty(default_network_constants.n_years).fill(np.nan)
                ),
                unit_max_capacity=pd.Series(
                    np.empty(default_network_constants.n_years).fill(np.nan)
                ),
                unit_min_capacity_increase=pd.Series(
                    np.empty(default_network_constants.n_years).fill(np.nan)
                ),
                unit_max_capacity_increase=pd.Series(
                    np.empty(default_network_constants.n_years).fill(np.nan)
                ),
            ),
            [
                NetworkValidatorException(
                    "Storage type must be of type StorageType, but it is <class 'str'> instead."
                )
            ],
        ),
    ],
    ids=["storage_type_is_none", "storage_type_is_not_StorageType"],
)
def test_validator_storage_type(
    storage_type: Any,
    storage: Storage,
    expected_exception_list: list[NetworkValidatorException],
) -> None:
    actual_exception_list: list[NetworkValidatorException] = []
    storage._validate_storage_type(storage_type, actual_exception_list)
    assert_same_exception_list(actual_exception_list, expected_exception_list)
