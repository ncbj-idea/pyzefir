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

from pyzefir.model.exceptions import NetworkValidatorException
from pyzefir.model.network import Network, NetworkElementsDict
from pyzefir.model.network_elements import Bus, Fuel, Generator, GeneratorType
from pyzefir.model.network_elements.emission_fee import EmissionFee
from tests.unit.defaults import (
    CO2_EMISSION,
    ELECTRICITY,
    HEATING,
    PM10_EMISSION,
    TRANSPORT,
    default_generator_type,
    default_generator_type_params,
    default_network_constants,
    get_default_generator_type,
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
    gen_type_a = get_default_generator_type(series_length=network.constants.n_years)
    gen_a = Generator(
        name="gen_A",
        energy_source_type=gen_type_a.name,
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
    network.add_generator_type(gen_type_a)
    network.add_generator(gen_a)
    network.add_emission_fee(emission_fee_A)
    network.add_emission_fee(emission_fee_B)
    return network


def test_if_all_validators_called(
    network: Network,
) -> None:
    generator = Generator(
        name="gen_A",
        energy_source_type=default_generator_type_params["name"],
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
        emission_fee={"EFee_A", "EFee_B"},
    )

    generator._validate_base_energy_source = MagicMock()
    generator._validate_generator_type = MagicMock()
    generator._validate_buses = MagicMock()
    generator._validate_emission_fee = MagicMock()

    generator.validate(network)

    generator._validate_base_energy_source.assert_called_once_with(
        network=network, exception_list=[]
    )
    generator._validate_generator_type.assert_called_once_with(
        exception_list=[],
        network=network,
        generator_type=network.generator_types.get(
            default_generator_type_params["name"]
        ),
    )
    generator._validate_buses.assert_called_once_with(
        exception_list=[],
        network=network,
        generator_type=network.generator_types.get(
            default_generator_type_params["name"]
        ),
    )
    generator._validate_emission_fee.assert_called_once_with(
        network=network, exception_list=[]
    )


def test_generator_validators(network: Network) -> None:
    gen = Generator(
        name="gen_I",
        energy_source_type="gen_type_non_instance",
        bus={"bus_A", "bus_B"},
        unit_base_cap=40,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
    )
    generator_type = 1
    network.generator_types["gen_type_non_instance"] = generator_type  # noqa
    actual_exception_list: list[NetworkValidatorException] = []
    expected_exception_list: list[NetworkValidatorException] = [
        NetworkValidatorException(
            "Generator type must be of type GeneratorType, but it is <class 'int'> instead."
        )
    ]
    gen._validate_generator_type(actual_exception_list, network, generator_type)  # noqa
    assert_same_exception_list(actual_exception_list, expected_exception_list)


def test_add_generator_to_network_without_generator_types(network: Network) -> None:
    # remove all existing generator types from the network
    network.generator_types = NetworkElementsDict()
    gen = Generator(
        name="gen_I",
        energy_source_type="gen_type_I",
        bus="bus_A",
        unit_base_cap=40,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
    )

    expected_exception_list: list[NetworkValidatorException] = [
        NetworkValidatorException("Network does not contain generator type")
    ]
    with pytest.raises(NetworkValidatorException) as e_info:
        gen.validate(network)
    assert_same_exception_list(list(e_info.value.exceptions), expected_exception_list)


def test_add_generator_with_unmatchable_energy_type(network: Network) -> None:
    gen_type = get_default_generator_type(
        series_length=network.constants.n_years,
        energy_types={HEATING},
        name="gen_type_c",
        conversion_rate={},
    )
    gen = Generator(
        name="gen_d",
        energy_source_type=gen_type.name,
        bus="bus_A",
        unit_base_cap=40,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
    )
    network.add_generator_type(gen_type)

    actual_exception_list: list[NetworkValidatorException] = []
    expected_exception_list: list[NetworkValidatorException] = [
        NetworkValidatorException(
            "Unable to attach generator to a bus bus_A. Bus energy type (ELECTRICITY) "
            "and generator energy types (['HEATING']) do not match"
        )
    ]
    gen._validate_buses(actual_exception_list, network, gen_type)
    assert_same_exception_list(actual_exception_list, expected_exception_list)


def test_add_generator_with_non_existing_bus_name(network: Network) -> None:
    gen_type = GeneratorType(**default_generator_type | {"energy_types": {ELECTRICITY}})
    gen_with_non_existing_bus = Generator(
        name="gen_B",
        energy_source_type=gen_type.name,
        bus="non_existing_bus_name",
        unit_base_cap=222,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
    )
    actual_exception_list: list[NetworkValidatorException] = []
    expected_exception_list: list[NetworkValidatorException] = [
        NetworkValidatorException(
            "Cannot attach generator to a bus non_existing_bus_name - bus "
            "does not exist in the network"
        )
    ]
    gen_with_non_existing_bus._validate_buses(actual_exception_list, network, gen_type)
    assert_same_exception_list(actual_exception_list, expected_exception_list)


def test_add_generator_with_wrong_energy_type(network: Network) -> None:
    gen_type = get_default_generator_type(
        series_length=network.constants.n_years,
        energy_types={TRANSPORT},
        name="gen_type_b",
    )
    network.add_generator_type(gen_type)
    gen = Generator(
        name="gen_c",
        energy_source_type="gen_type_b",
        bus={"bus_A", "bus_B"},
        unit_base_cap=10,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
    )
    expected_exception_list: list[NetworkValidatorException] = [
        NetworkValidatorException(
            "Gen energy types: ['TRANSPORT'] are not "
            "compliant with the network energy types: ['ELECTRICITY', 'HEATING']"
        )
    ]
    with pytest.raises(NetworkValidatorException) as e_info:
        gen.validate(network)
    assert_same_exception_list(list(e_info.value.exceptions), expected_exception_list)


def test_generator_is_connected_to_correct_bus(network: Network) -> None:
    test_gen = Generator(
        name="test_gen",
        energy_source_type=default_generator_type_params["name"],
        bus="bus_A",  # missing bus_B
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
    with pytest.raises(NetworkValidatorException) as e_info:
        test_gen.validate(network)

    assert_same_exception_list(
        list(e_info.value.exceptions),
        [
            NetworkValidatorException(
                "Conversion_rate for energy types: "
                "['HEATING'] which are not in connected buses energy types"
            )
        ],
    )


@pytest.mark.parametrize(
    "gen_type, gen, expected_exception_list",
    [
        (
            get_default_generator_type(
                series_length=default_network_constants.n_years,
                energy_types={ELECTRICITY},
                name="gen_type_c",
            ),
            Generator(
                name="gen_c",
                energy_source_type="gen_type_c",
                bus=[],  # noqa
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
            [
                NetworkValidatorException(
                    "Generator attribute 'buses' for gen_c must be an instance of <class 'set'>, "
                    "but it is an instance of <class 'list'> instead"
                ),
                NetworkValidatorException(
                    "Conversion_rate for energy types: ['ELECTRICITY', 'HEATING'] "
                    "which are not in connected buses energy types"
                ),
            ],
        ),
        (
            get_default_generator_type(
                series_length=default_network_constants.n_years,
                energy_types={ELECTRICITY},
                name="gen_type_d",
            ),
            Generator(
                name="gen_d",
                energy_source_type="gen_type_c",
                bus={1, 2.0},  # noqa
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
            [
                NetworkValidatorException(
                    "Generator attribute 'buses' must contain only strings"
                ),
                NetworkValidatorException(
                    "Cannot attach generator to a bus 1 - bus does not exist in the network"
                ),
                NetworkValidatorException(
                    "Cannot attach generator to a bus 2.0 - bus does not exist in the network"
                ),
            ],
        ),
        (
            get_default_generator_type(series_length=default_network_constants.n_years),
            Generator(
                name="gen_d",
                energy_source_type="gen_type_c",
                bus={"bus_A", "bus_B"},
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
            [],
        ),
    ],
    ids=[
        "test_if_buses_attr_is_not_set",
        "test_if_buses_attr_doesnt_contain_strings",
        "test_buses_ok",
    ],
)
def test_generator_validators_if_buses_not_set_or_not_contain_strings(
    network: Network,
    gen_type: GeneratorType,
    gen: Generator,
    expected_exception_list: list[NetworkValidatorException],
) -> None:
    actual_exception_list: list[NetworkValidatorException] = []
    gen._validate_buses(actual_exception_list, network, gen_type)
    assert_same_exception_list(actual_exception_list, expected_exception_list)


def test_validate_emission_fee_validator(network: Network) -> None:
    generator = Generator(
        name="gen_A",
        energy_source_type=default_generator_type_params["name"],
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
        emission_fee={"New_Emission_Fee"},
    )
    with pytest.raises(NetworkValidatorException) as e_info:
        generator.validate(network)

    assert_same_exception_list(
        list(e_info.value.exceptions),
        [
            NetworkValidatorException(
                "Network does not contain Emission Fee: New_Emission_Fee in its structure"
            )
        ],
    )

    emission_fee_c = EmissionFee(
        name="EFee_C",
        emission_type=CO2_EMISSION,
        price=pd.Series([0.4] * default_network_constants.n_years),
    )
    network.add_emission_fee(emission_fee_c)
    emission_fee_d = EmissionFee(
        name="EFee_D",
        emission_type=CO2_EMISSION,
        price=pd.Series([0.4] * default_network_constants.n_years),
    )
    network.add_emission_fee(emission_fee_d)
    generator.emission_fee = {"EFee_A", "EFee_C", "EFee_D"}

    with pytest.raises(NetworkValidatorException) as e_info:
        generator.validate(network)

    assert_same_exception_list(
        list(e_info.value.exceptions),
        [
            NetworkValidatorException(
                "There are fees: ['EFee_A', 'EFee_C', 'EFee_D'] which apply "
                "to the same type of emission: CO2"
            )
        ],
    )


def test_add_generator_buses_same_energy_type(network: Network) -> None:
    gen_type = get_default_generator_type(
        series_length=network.constants.n_years,
        energy_types={ELECTRICITY},
        name="gen_type_b",
    )
    network.add_generator_type(gen_type)
    gen = Generator(
        name="gen_c",
        energy_source_type="gen_type_b",
        bus={"bus_A", "bus_B"},
        unit_base_cap=10,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
    )
    gen.validate(network)
    bus_c = Bus(name="bus_C", energy_type=ELECTRICITY)
    network.add_bus(bus_c)
    gen.buses = {"bus_A", "bus_B", "bus_C"}
    expected_exception_list: list[NetworkValidatorException] = [
        NetworkValidatorException(
            "Buses ['bus_A', 'bus_C'] have the same energy_type ELECTRICITY which is not allowed"
        )
    ]
    with pytest.raises(NetworkValidatorException) as e_info:
        gen.validate(network)
    assert_same_exception_list(list(e_info.value.exceptions), expected_exception_list)
