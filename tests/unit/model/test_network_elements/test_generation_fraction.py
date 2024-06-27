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

from pyzefir.model.exceptions import NetworkValidatorException
from pyzefir.model.network import Network
from pyzefir.model.network_elements.bus import Bus
from pyzefir.model.network_elements.energy_sources.generator import Generator
from pyzefir.model.network_elements.energy_sources.storage import Storage
from pyzefir.model.network_elements.fuel import Fuel
from pyzefir.model.network_elements.generation_fraction import GenerationFraction
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


@pytest.fixture()
def generation_fraction_attrs() -> dict[str, str | pd.Series]:
    return {
        "name": "test_gf",
        "tag": "tag_gen_A",
        "sub_tag": "tag_stor_A",
        "fraction_type": "hourly",
        "energy_type": ELECTRICITY,
        "min_generation_fraction": pd.Series([np.nan, 0, np.nan, 0]),
        "max_generation_fraction": pd.Series([np.nan, 0.8, np.nan, 0.9]),
    }


@pytest.fixture()
def network() -> Network:
    network = Network(
        emission_types=[CO2_EMISSION, PM10_EMISSION],
        energy_types=[ELECTRICITY, HEATING],
        network_constants=default_network_constants,
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
        unit_min_capacity=pd.Series([np.nan] * network.constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * network.constants.n_years),
        unit_min_capacity_increase=pd.Series([np.nan] * network.constants.n_years),
        unit_max_capacity_increase=pd.Series([np.nan] * network.constants.n_years),
        tags=["tag_gen_A"],
    )
    gen_b = Generator(
        name="gen_B",
        energy_source_type=gen_type.name,
        bus={"bus_A", "bus_B"},
        unit_base_cap=25,
        unit_min_capacity=pd.Series([np.nan] * network.constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * network.constants.n_years),
        unit_min_capacity_increase=pd.Series([np.nan] * network.constants.n_years),
        unit_max_capacity_increase=pd.Series([np.nan] * network.constants.n_years),
        tags=["tag_gen_B"],
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
        tags=["tag_stor_A"],
    )
    network.add_bus(bus_a)
    network.add_bus(bus_b)
    network.add_generator_type(gen_type)
    network.add_storage_type(stor_type)
    network.add_generator(gen_a)
    network.add_generator(gen_b)
    network.add_storage(stor_a)

    return network


def test_generation_fraction_validation(
    network: Network, generation_fraction_attrs: dict[str, str | pd.Series]
) -> None:
    gf = GenerationFraction(**generation_fraction_attrs)
    with patch.object(gf, "_validate_types") as mock_validate_types, patch.object(
        gf, "_validate_energy_type"
    ) as mock_validate_energy_type, patch.object(
        gf, "_validate_fraction_series"
    ) as mock_validate_fraction_series, patch.object(
        gf, "_validate_tags"
    ) as mock_validate_tags:

        gf.validate(network)

        mock_validate_types.assert_called_once_with([])
        mock_validate_energy_type.assert_called_once_with(network, [])
        mock_validate_fraction_series.assert_called_once_with(network, [])
        mock_validate_tags.assert_called_once_with(network, [])


@pytest.mark.parametrize(
    "invalid_attr, error_msg",
    [
        pytest.param(
            {
                "tag": 12,
            },
            "GenerationFraction attribute 'tag' for test_gf must be an instance of <class 'str'>, "
            "but it is an instance of <class 'int'> instead",
            id="tag_int",
        ),
        pytest.param(
            {
                "fraction_type": ("hourly",),
            },
            "GenerationFraction attribute 'fraction_type' for test_gf must be an instance of <class 'str'>, "
            "but it is an instance of <class 'tuple'> instead",
            id="fraction_type_tuple",
        ),
        pytest.param(
            {
                "energy_type": [ELECTRICITY],
            },
            "GenerationFraction attribute 'energy_type' for test_gf must be an instance of <class 'str'>, "
            "but it is an instance of <class 'list'> instead",
            id="energy_type_list",
        ),
        pytest.param(
            {
                "min_generation_fraction": pd.DataFrame(data=[np.nan, 0, np.nan, 0]),
            },
            "GenerationFraction attribute 'min_generation_fraction' for test_gf must be an instance of "
            "<class 'pandas.core.series.Series'>, "
            "but it is an instance of <class 'pandas.core.frame.DataFrame'> instead",
            id="min_generation_fraction_dataframe",
        ),
    ],
)
def test_generation_fraction_validate_types(
    invalid_attr: dict[str, Any],
    generation_fraction_attrs: dict[str, str | pd.Series],
    error_msg: str,
) -> None:
    generation_fraction = GenerationFraction(
        **(generation_fraction_attrs | invalid_attr)
    )
    with pytest.raises(NetworkValidatorException) as e_info:
        generation_fraction._validate_types([])
    assert str(e_info.value) == error_msg


@pytest.mark.parametrize(
    "energy_type, exception_list",
    [
        pytest.param(
            HEATING,
            [],
            id="happy_path",
        ),
        pytest.param(
            "COLD",
            [
                NetworkValidatorException(
                    "Energy type COLD not found in the Network energy types: ['ELECTRICITY', 'HEATING']"
                )
            ],
            id="et_cold",
        ),
        pytest.param(
            "TEST_ENERGY_TYPE",
            [
                NetworkValidatorException(
                    "Energy type TEST_ENERGY_TYPE not found in the Network energy types: ['ELECTRICITY', 'HEATING']"
                )
            ],
            id="et_cold",
        ),
    ],
)
def test_generation_fraction_validate_energy_type(
    network: Network,
    energy_type: str,
    generation_fraction_attrs: dict[str, str | pd.Series],
    exception_list: list[NetworkValidatorException],
) -> None:
    generation_fraction = GenerationFraction(
        **generation_fraction_attrs | {"energy_type": energy_type}
    )
    actual_exception_list: list[NetworkValidatorException] = []
    generation_fraction._validate_energy_type(network, actual_exception_list)
    assert_same_exception_list(actual_exception_list, exception_list)


@pytest.mark.parametrize(
    "fraction_series, exception_list",
    [
        pytest.param(
            pd.Series([np.nan, 0.2, np.nan, 0.2]),
            [],
            id="happy_path",
        ),
        pytest.param(
            pd.Series([np.nan, 0.2]),
            [NetworkValidatorException("min_generation_fraction must have 4 values")],
            id="series_not_in_years_range",
        ),
        pytest.param(
            pd.Series([1, 1, 0, 0]),
            [
                NetworkValidatorException(
                    "min_generation_fraction type is int64 but should be float"
                )
            ],
            id="series_contains_int",
        ),
        pytest.param(
            pd.Series([1.2, 1.1, -0.4, 0]),
            [
                NetworkValidatorException(
                    "Invalid fraction series 'min_generation_fraction': all passed values must be between 0.0 and 1.0"
                )
            ],
            id="series_values_not_in_range(0,1)",
        ),
    ],
)
def test_generation_fraction_validate_fraction_series(
    network: Network,
    fraction_series: pd.Series,
    generation_fraction_attrs: dict[str, str | pd.Series],
    exception_list: list[NetworkValidatorException],
) -> None:
    generation_fraction = GenerationFraction(
        **generation_fraction_attrs | {"min_generation_fraction": fraction_series}
    )
    actual_exception_list: list[NetworkValidatorException] = []
    generation_fraction._validate_fraction_series(network, actual_exception_list)
    assert_same_exception_list(actual_exception_list, exception_list)


@pytest.mark.parametrize(
    "tag_name, exception_list",
    [
        pytest.param(
            "tag_gen_A",
            [],
            id="happy_path",
        ),
        pytest.param(
            "TEST_TAG",
            [
                NetworkValidatorException(
                    "Provided tag TEST_TAG not found in any generator or storage in Network."
                )
            ],
            id="TEST_TAG_not_in_network",
        ),
        pytest.param(
            "MORE_TEST_TAG",
            [
                NetworkValidatorException(
                    "Provided tag MORE_TEST_TAG not found in any generator or storage in Network."
                )
            ],
            id="MORE_TEST_TAG_not_in_network",
        ),
    ],
)
def test_generation_fraction_validate_tag(
    network: Network,
    tag_name: str,
    generation_fraction_attrs: dict[str, str | pd.Series],
    exception_list: list[NetworkValidatorException],
) -> None:
    generation_fraction = GenerationFraction(
        **generation_fraction_attrs | {"tag": tag_name}
    )
    actual_exception_list: list[NetworkValidatorException] = []
    generation_fraction._validate_tags(network, actual_exception_list)
    assert_same_exception_list(actual_exception_list, exception_list)


def test_generation_fraction_validate_tags_energy_type(
    network: Network, generation_fraction_attrs: dict[str, str | pd.Series]
) -> None:
    generation_fraction = GenerationFraction(
        **generation_fraction_attrs | {"energy_type": "COLD"}
    )
    actual_exception_list: list[NetworkValidatorException] = []
    exception_list: list[NetworkValidatorException] = [
        NetworkValidatorException(
            "Energy type COLD do not match to generator or storage in Network related with tag_gen_A or tag_stor_A"
        )
    ]
    generation_fraction._validate_tags(network, actual_exception_list)
    print(actual_exception_list)
    assert_same_exception_list(actual_exception_list, exception_list)


def test_generation_fraction_add_to_network(
    network: Network, generation_fraction_attrs: dict[str, str | pd.Series]
) -> None:
    generation_fraction = GenerationFraction(**generation_fraction_attrs)
    network.add_generation_fraction(generation_fraction)

    assert network.generation_fractions

    with pytest.raises(NetworkValidatorException) as error:
        network.add_generation_fraction(None)
    assert error.value.args[0] == "Generation Fraction cannot be None"

    generation_fraction.energy_type = "ULTRA_COLD"
    generation_fraction.name = "TEST_CASE_WRONG_ET"
    with pytest.raises(NetworkValidatorException) as error:
        network.add_generation_fraction(generation_fraction)
    assert (
        error.value.args[0]
        == "While adding generation fraction TEST_CASE_WRONG_ET following errors occurred: "
    )
    assert (
        error.value.args[1][0].args[0]
        == "Energy type ULTRA_COLD not found in the Network energy types: ['ELECTRICITY', 'HEATING']"
    )
