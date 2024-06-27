# PyZefir
# Copyright (C) 2024 Narodowe Centrum Badań Jądrowych
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
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network import Network
from pyzefir.model.network_elements import Bus, EmissionFee, Fuel, Generator, Storage
from pyzefir.model.network_elements.capacity_bound import CapacityBound
from tests.unit.defaults import (
    CO2_EMISSION,
    ELECTRICITY,
    HEATING,
    PM10_EMISSION,
    default_network_constants,
    get_default_generator_type,
    get_default_storage_type,
)
from tests.unit.model.test_network_elements.helpers import assert_same_exception_list


@pytest.fixture
def network() -> Network:
    network = Network(
        energy_types=[ELECTRICITY, HEATING],
        network_constants=default_network_constants,
        emission_types=[CO2_EMISSION, PM10_EMISSION],
    )
    coal = Fuel(
        name="coal",
        cost=pd.Series([1, 2, 3, 4]),
        availability=pd.Series([1, 2, 3, 4]),
        emission={},
        energy_per_unit=0.4,
    )
    network.add_fuel(coal)
    bus_a = Bus(name="bus_A", energy_type=ELECTRICITY)
    bus_b = Bus(name="bus_B", energy_type=HEATING)
    gen_type = get_default_generator_type(series_length=network.constants.n_years)
    stor_type = get_default_storage_type(series_length=network.constants.n_years)
    gen_a = Generator(
        name="gen_A",
        energy_source_type=gen_type.name,
        bus={"bus_A", "bus_B"},
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
    gen_b = Generator(
        name="gen_B",
        energy_source_type=gen_type.name,
        bus={"bus_A", "bus_B"},
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
    stor_a = Storage(
        name="stor_a",
        energy_source_type=stor_type.name,
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
    emission_fee_A = EmissionFee(
        name="EFee_A",
        emission_type=CO2_EMISSION,
        price=pd.Series([0.1] * default_network_constants.n_years),
    )
    emission_fee_B = EmissionFee(
        name="EFee_B",
        emission_type=PM10_EMISSION,
        price=pd.Series([0.1] * default_network_constants.n_years),
    )

    network.add_bus(bus_a)
    network.add_bus(bus_b)
    network.add_generator_type(gen_type)
    network.add_storage_type(stor_type)
    network.add_generator(gen_a)
    network.add_generator(gen_b)
    network.add_storage(stor_a)
    network.add_emission_fee(emission_fee_A)
    network.add_emission_fee(emission_fee_B)
    return network


def test_capacity_bound_validation(network: Network) -> None:
    capacity_bound = CapacityBound(
        name="test_cb",
        left_technology="gen_left",
        right_technology="gen_right",
        sense="EQ",
        left_coefficient=0.4,
    )

    with patch.object(
        capacity_bound, "_validate_fields_types"
    ) as mock_validate_fields_types, patch.object(
        capacity_bound, "_validate_technology"
    ) as mock_validate_technology, patch.object(
        capacity_bound, "_validate_sense_value"
    ) as mock_validate_sense_value, patch.object(
        capacity_bound, "_validate_left_coefficient"
    ) as mock_validate_left_coefficient:

        capacity_bound.validate(network)

        mock_validate_fields_types.assert_called_once_with([])
        mock_validate_technology.assert_called_once_with(network, [])
        mock_validate_sense_value.assert_called_once_with([])
        mock_validate_left_coefficient.assert_called_once_with([])


@pytest.mark.parametrize(
    "attrs, error_msg",
    [
        pytest.param(
            {
                "name": "test_cb",
                "left_technology": 12,
                "right_technology": "gen_right",
                "sense": "EQ",
                "left_coefficient": 0.4,
            },
            "CapacityBound attribute 'left_technology' for test_cb must be an instance of <class 'str'>, "
            "but it is an instance of <class 'int'> instead",
            id="left_technology_int",
        ),
        pytest.param(
            {
                "name": "test_cb",
                "left_technology": "gen_left",
                "right_technology": ["gen_right"],
                "sense": "EQ",
                "left_coefficient": 0.4,
            },
            "CapacityBound attribute 'right_technology' for test_cb must be an instance of <class 'str'>, "
            "but it is an instance of <class 'list'> instead",
            id="right_technology_list",
        ),
        pytest.param(
            {
                "name": "test_cb",
                "left_technology": "gen_left",
                "right_technology": "gen_right",
                "sense": 1.24,
                "left_coefficient": 0.4,
            },
            "CapacityBound attribute 'sense' for test_cb must be an instance of <class 'str'>, "
            "but it is an instance of <class 'float'> instead",
            id="sense_float",
        ),
        pytest.param(
            {
                "name": "test_cb",
                "left_technology": "gen_left",
                "right_technology": "gen_right",
                "sense": "EQ",
                "left_coefficient": 1,
            },
            "CapacityBound attribute 'left_coefficient' for test_cb must be an instance of <class 'float'>, "
            "but it is an instance of <class 'int'> instead",
            id="left_coefficient_int",
        ),
    ],
)
def test_capacity_bound_validate_types(attrs: dict[str, Any], error_msg: str) -> None:
    capacity_bound = CapacityBound(**attrs)
    with pytest.raises(NetworkValidatorException) as e_info:
        capacity_bound._validate_fields_types([])
    assert str(e_info.value) == error_msg


@pytest.mark.parametrize(
    "capacity_bound, exception_list",
    [
        pytest.param(
            CapacityBound(
                name="test_cb",
                left_technology="gen_A",
                right_technology="gen_B",
                sense="EQ",
                left_coefficient=0.4,
            ),
            [],
            id="happy_path_both_gens",
        ),
        pytest.param(
            CapacityBound(
                name="test_cb",
                left_technology="gen_A",
                right_technology="stor_a",
                sense="EQ",
                left_coefficient=0.4,
            ),
            [],
            id="happy_path_gen_and_stor",
        ),
        pytest.param(
            CapacityBound(
                name="test_cb",
                left_technology="Generator",
                right_technology="gen_B",
                sense="EQ",
                left_coefficient=0.4,
            ),
            [
                NetworkValidatorException(
                    "Technology name 'Generator' is not present in network generators or storages."
                )
            ],
            id="left_technology_not_in_gens",
        ),
        pytest.param(
            CapacityBound(
                name="test_cb",
                left_technology="Generator",
                right_technology="MorePowerGenerator",
                sense="EQ",
                left_coefficient=0.4,
            ),
            [
                NetworkValidatorException(
                    "Technology name 'Generator' is not present in network generators or storages."
                ),
                NetworkValidatorException(
                    "Technology name 'MorePowerGenerator' is not present in network generators or storages."
                ),
            ],
            id="both_gens_not_in_network",
        ),
        pytest.param(
            CapacityBound(
                name="test_cb",
                left_technology="stor_a",
                right_technology="Storage",
                sense="EQ",
                left_coefficient=0.4,
            ),
            [
                NetworkValidatorException(
                    "Technology name 'Storage' is not present in network generators or storages."
                ),
            ],
            id="right_technology_not_in_storages",
        ),
        pytest.param(
            CapacityBound(
                name="test_cb",
                left_technology="BetterStorage",
                right_technology="Storage",
                sense="EQ",
                left_coefficient=0.4,
            ),
            [
                NetworkValidatorException(
                    "Technology name 'BetterStorage' is not present in network generators or storages."
                ),
                NetworkValidatorException(
                    "Technology name 'Storage' is not present in network generators or storages."
                ),
            ],
            id="both_stors_not_in_storages",
        ),
    ],
)
def test_capacity_bound_validate_technology(
    network: Network,
    capacity_bound: CapacityBound,
    exception_list: list[NetworkValidatorException],
) -> None:
    actual_exception_list: list[NetworkValidatorException] = []
    capacity_bound._validate_technology(network, actual_exception_list)
    assert_same_exception_list(actual_exception_list, exception_list)


@pytest.mark.parametrize(
    "capacity_bound, exception_list",
    [
        pytest.param(
            CapacityBound(
                name="test_cb",
                left_technology="gen_A",
                right_technology="gen_B",
                sense="EQ",
                left_coefficient=0.4,
            ),
            [],
            id="happy_path_EQ",
        ),
        pytest.param(
            CapacityBound(
                name="test_cb",
                left_technology="gen_A",
                right_technology="gen_B",
                sense="LEQ",
                left_coefficient=0.4,
            ),
            [],
            id="happy_path_LEQ",
        ),
        pytest.param(
            CapacityBound(
                name="test_cb",
                left_technology="gen_A",
                right_technology="gen_B",
                sense="eq",
                left_coefficient=0.4,
            ),
            [
                NetworkValidatorException(
                    "The provided sense 'eq' is not valid. Valid senses are: ['EQ', 'LEQ']."
                )
            ],
            id="eq_lower_case",
        ),
        pytest.param(
            CapacityBound(
                name="test_cb",
                left_technology="gen_A",
                right_technology="gen_B",
                sense="SENSE",
                left_coefficient=0.4,
            ),
            [
                NetworkValidatorException(
                    "The provided sense 'SENSE' is not valid. Valid senses are: ['EQ', 'LEQ']."
                )
            ],
            id="sense_different",
        ),
    ],
)
def test_capacity_bound_validate_sense(
    capacity_bound: CapacityBound,
    exception_list: list[NetworkValidatorException],
) -> None:
    actual_exception_list: list[NetworkValidatorException] = []
    capacity_bound._validate_sense_value(actual_exception_list)
    assert_same_exception_list(actual_exception_list, exception_list)


@pytest.mark.parametrize(
    "capacity_bound, exception_list",
    [
        pytest.param(
            CapacityBound(
                name="test_cb",
                left_technology="gen_A",
                right_technology="gen_B",
                sense="EQ",
                left_coefficient=0.4,
            ),
            [],
            id="happy_path",
        ),
        pytest.param(
            CapacityBound(
                name="test_cb",
                left_technology="gen_A",
                right_technology="gen_B",
                sense="EQ",
                left_coefficient=0.0,
            ),
            [],
            id="happy_path_lower_edge",
        ),
        pytest.param(
            CapacityBound(
                name="test_cb",
                left_technology="gen_A",
                right_technology="gen_B",
                sense="EQ",
                left_coefficient=1.0,
            ),
            [],
            id="happy_path_upper_edge",
        ),
        pytest.param(
            CapacityBound(
                name="test_cb",
                left_technology="gen_A",
                right_technology="gen_B",
                sense="EQ",
                left_coefficient=1.1,
            ),
            [
                NetworkValidatorException(
                    "The provided left coefficient '1.1' is not valid. It must be between <0.0 and 1.0>."
                )
            ],
            id="coeff above 1",
        ),
        pytest.param(
            CapacityBound(
                name="test_cb",
                left_technology="gen_A",
                right_technology="gen_B",
                sense="EQ",
                left_coefficient=-0.1,
            ),
            [
                NetworkValidatorException(
                    "The provided left coefficient '-0.1' is not valid. It must be between <0.0 and 1.0>."
                )
            ],
            id="coeff negative",
        ),
    ],
)
def test_capacity_bound_validate_left_coefficient(
    capacity_bound: CapacityBound,
    exception_list: list[NetworkValidatorException],
) -> None:
    actual_exception_list: list[NetworkValidatorException] = []
    capacity_bound._validate_left_coefficient(actual_exception_list)
    assert_same_exception_list(actual_exception_list, exception_list)


@pytest.mark.parametrize(
    "capacity_bound, exception_list",
    [
        pytest.param(
            CapacityBound(
                name="test_cb",
                left_technology="Gens",
                right_technology="gen_B",
                sense="EQ",
                left_coefficient=0.0,
            ),
            [
                NetworkValidatorException(
                    "Technology name 'Gens' is not present in network generators or storages."
                )
            ],
            id="left_technology_not_in_network",
        ),
        pytest.param(
            CapacityBound(
                name="test_cb",
                left_technology="Generator_1",
                right_technology="Generator_2",
                sense="REQ",
                left_coefficient=1.24,
            ),
            [
                NetworkValidatorException(
                    "Technology name 'Generator_1' is not present in network generators or storages."
                ),
                NetworkValidatorException(
                    "Technology name 'Generator_2' is not present in network generators or storages."
                ),
                NetworkValidatorException(
                    "The provided sense 'REQ' is not valid. Valid senses are: ['EQ', 'LEQ']."
                ),
                NetworkValidatorException(
                    "The provided left coefficient '1.24' is not valid. It must be between <0.0 and 1.0>."
                ),
            ],
            id="all_attributes_with_errors",
        ),
    ],
)
def test_capacity_bound_validate(
    network: Network,
    capacity_bound: CapacityBound,
    exception_list: list[NetworkValidatorException],
) -> None:
    with pytest.raises(NetworkValidatorExceptionGroup) as e_info:
        capacity_bound.validate(network)
    assert_same_exception_list(list(e_info.value.exceptions), exception_list)


def test_capacity_bound_add_to_network(
    network: Network,
) -> None:
    capacity_bound = CapacityBound(
        name="test_cb",
        left_technology="gen_A",
        right_technology="gen_B",
        sense="EQ",
        left_coefficient=0.4,
    )
    network.add_capacity_bound(capacity_bound)
    assert network.capacity_bounds
    with pytest.raises(NetworkValidatorException) as error:
        network.add_capacity_bound(None)

    assert error.value.args[0] == "CapacityBound cannot be None"
