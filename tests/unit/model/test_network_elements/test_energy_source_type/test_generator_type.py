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
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.exceptions import NetworkValidatorException
from pyzefir.model.network import Network
from pyzefir.model.network_elements import GeneratorType
from pyzefir.model.network_elements.bus import Bus
from pyzefir.model.network_elements.energy_source_types.generator_type import (
    SumNotEqualToOneWarning,
)
from pyzefir.model.network_elements.energy_sources.generator import Generator
from pyzefir.model.network_elements.fuel import Fuel
from tests.unit.defaults import (
    CO2_EMISSION,
    DEFAULT_HOURS,
    DEFAULT_YEARS,
    ELECTRICITY,
    HEATING,
    default_generator_type,
    default_generator_type_netto,
    default_network_constants,
    default_network_constants_netto_cost,
    get_default_generator_type,
)
from tests.unit.model.test_network_elements.helpers import assert_same_exception_list


@pytest.fixture
def network(request: pytest.FixtureRequest) -> Network:
    if request.node.get_closest_marker("netto_cost"):
        network_constants = default_network_constants_netto_cost
    else:
        network_constants = default_network_constants
    network = Network(
        energy_types=[ELECTRICITY, HEATING],
        network_constants=network_constants,
        emission_types=["CO2", "PM10"],
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
        unit_min_capacity=pd.Series([np.nan] * network.constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * network.constants.n_years),
        unit_min_capacity_increase=pd.Series([np.nan] * network.constants.n_years),
        unit_max_capacity_increase=pd.Series([np.nan] * network.constants.n_years),
    )
    network.add_bus(bus_a)
    network.add_bus(bus_b)
    network.add_generator_type(gen_type_a)
    network.add_generator(gen_a)
    return network


def test_capacity_bound_validation(network: Network) -> None:
    gen_type = get_default_generator_type(
        series_length=default_network_constants.n_years,
    )

    with patch.object(gen_type, "_validate_fuels") as mock_validate_fuels, patch.object(
        gen_type, "_validate_capacity_factor"
    ) as mock_validate_capacity_factor, patch.object(
        gen_type, "_validate_efficiency"
    ) as mock_validate_efficiency, patch.object(
        gen_type, "_validate_emission_reduction"
    ) as mock_validate_emission_reduction, patch.object(
        gen_type, "_validate_conversion_rate"
    ) as mock_validate_conversion_rate, patch.object(
        gen_type, "_validate_power_utilization_boundaries"
    ) as mock_validate_power_utilization, patch.object(
        gen_type, "_validate_generation_compensation"
    ) as mock_validate_generation_compensation, patch.object(
        gen_type, "_validate_ramp"
    ) as mock_validate_ramp:

        gen_type.validate(network)

        mock_validate_fuels.assert_called_once_with([], network)
        mock_validate_capacity_factor.assert_called_once_with([], network)
        mock_validate_efficiency.assert_called_once_with([], network)
        mock_validate_emission_reduction.assert_called_once_with([], network)
        mock_validate_conversion_rate.assert_called_once_with([], network)
        mock_validate_power_utilization.assert_called_once_with(network, [])
        mock_validate_generation_compensation.assert_called_once_with([])
        mock_validate_ramp.assert_called_once_with([])


@pytest.mark.parametrize(
    "gen_type, exception_list",
    [
        (
            get_default_generator_type(
                series_length=default_network_constants.n_years, ramp_down="sadasd"
            ),
            [NetworkValidatorException("ramp_down value must be float or empty.")],
        ),
        (
            get_default_generator_type(
                series_length=default_network_constants.n_years,
                name="gen_type_cap_fuel",
                capacity_factor="dummy_cap_f",
            ),
            [
                NetworkValidatorException(
                    "Generator type can have either capacity factor or fuel at the same time"
                ),
                NetworkValidatorException(
                    "Capacity factor 'dummy_cap_f' has not been added to the network"
                ),
            ],
        ),
        (
            get_default_generator_type(
                series_length=default_network_constants.n_years,
                name="gen_type_cap_fuel",
                ramp_up=1,
            ),
            [
                NetworkValidatorException(
                    "ramp_up value must be greater than 0 and less than 1, but it is 1"
                ),
            ],
        ),
    ],
)
def test_generator_type_validate(
    gen_type: GeneratorType,
    exception_list: list[NetworkValidatorException],
    network: Network,
) -> None:
    with pytest.raises(NetworkValidatorException) as e_info:
        gen_type.validate(network)
    assert_same_exception_list(list(e_info.value.exceptions), exception_list)


@pytest.mark.parametrize(
    "gen_type, exception_list",
    [
        (
            GeneratorType(**default_generator_type | {"fuel": 1, "name": "gen_type_E"}),
            [
                NetworkValidatorException(
                    "None or str type for fuel expected but type: <class 'int'> given"
                ),
                NetworkValidatorException("Fuel 1 has not been added to the network"),
            ],
        ),
    ],
)
def test_fuel_validators(
    gen_type: GeneratorType,
    exception_list: list[NetworkValidatorException],
    network: Network,
) -> None:
    actual_exception_list: list[NetworkValidatorException] = []
    gen_type._validate_fuels(actual_exception_list, network)
    assert_same_exception_list(actual_exception_list, exception_list)


@pytest.mark.parametrize(
    "gen_type, exception_list",
    [
        (
            GeneratorType(
                **default_generator_type
                | {"name": "gen_type_F", "capacity_factor": 1, "fuel": None}
            ),
            [
                NetworkValidatorException(
                    "None or str type for capacity factor expected but type: <class 'int'> given"
                ),
                NetworkValidatorException(
                    "Capacity factor '1' has not been added to the network"
                ),
            ],
        ),
        (
            GeneratorType(
                **default_generator_type
                | {"name": "gen_type_I", "capacity_factor": "test_gen_I", "fuel": None}
            ),
            [
                NetworkValidatorException(
                    "Capacity factor 'test_gen_I' has not been added to the network"
                ),
            ],
        ),
    ],
)
def test_capacity_factor_validators(
    gen_type: GeneratorType,
    exception_list: list[NetworkValidatorException],
    network: Network,
) -> None:
    actual_exception_list: list[NetworkValidatorException] = []
    gen_type._validate_capacity_factor(actual_exception_list, network)
    assert_same_exception_list(actual_exception_list, exception_list)


@pytest.mark.parametrize(
    "gen_type, exception_list",
    [
        pytest.param(
            GeneratorType(
                **default_generator_type
                | {
                    "name": "gen_type_L",
                    "efficiency": pd.DataFrame(
                        {"DUMMY_ELECTRICITY": [0.9], "DUMMY_HEATING": [0.6]}
                    ),
                }
            ),
            [
                NetworkValidatorException(
                    "Efficiency energy types do not exist in network energy types: "
                    "['ELECTRICITY', 'HEATING']"
                )
            ],
            id="Energy types not in network",
        ),
        pytest.param(
            GeneratorType(
                **default_generator_type
                | {"name": "gen_type_no_eff", "efficiency": None}
            ),
            [NetworkValidatorException("Efficiency cannot be None.")],
            id="efficiency is None",
        ),
        pytest.param(
            GeneratorType(
                **default_generator_type_netto
                | {
                    "name": "gen_type_netto",
                    "efficiency": pd.DataFrame(
                        {"ELECTRICITY": [0.9], "HEATING": [0.6]}
                    ),
                }
            ),
            [
                NetworkValidatorException(
                    "In generator type: gen_type_netto generator capacity cost is set to netto which required "
                    "efficiency only for one energy type: {'ELECTRICITY'} but efficiency has been defined for "
                    "['ELECTRICITY', 'HEATING']"
                )
            ],
            marks=pytest.mark.netto_cost,
            id="Capacity cost netto but 2 energy types provided",
        ),
    ],
)
def test_efficiency_validators(
    gen_type: GeneratorType,
    exception_list: list[NetworkValidatorException],
    network: Network,
) -> None:
    actual_exception_list: list[NetworkValidatorException] = []
    gen_type._validate_efficiency(actual_exception_list, network)
    assert_same_exception_list(actual_exception_list, exception_list)


@pytest.mark.parametrize(
    "gen_type, exception_list",
    [
        pytest.param(
            GeneratorType(
                **default_generator_type
                | {
                    "name": "gen_type_no_e_r",
                    "emission_reduction": "Emission_reduction",
                }
            ),
            [
                NetworkValidatorException(
                    "Emission reduction must be type: dict[str, pd.Series] but it's: <class 'str'>."
                )
            ],
            id="type not dict[str, pd.Series]",
        ),
        pytest.param(
            GeneratorType(
                **default_generator_type
                | {
                    "name": "gen_type_non_existed_e_r",
                    "emission_reduction": {
                        "DUMMY_E_R": pd.Series([0.9] * DEFAULT_YEARS)
                    },
                }
            ),
            [
                NetworkValidatorException(
                    "Emission reduction emission types {'DUMMY_E_R'} do not exist in network "
                    "emission types: ['CO2', 'PM10']"
                )
            ],
            id="Emission type not in network emission types",
        ),
        pytest.param(
            GeneratorType(
                **default_generator_type
                | {
                    "name": "gen_type_non_existed_e_r",
                    "emission_reduction": {CO2_EMISSION: pd.Series([0.9] * 2)},
                }
            ),
            [
                NetworkValidatorException(
                    "gen_type_non_existed_e_r Emission reduction must have 4 values"
                )
            ],
            id="Emission type not in n_years range",
        ),
    ],
)
def test_emission_reduction_validators(
    gen_type: GeneratorType,
    exception_list: list[NetworkValidatorException],
    network: Network,
) -> None:
    actual_exception_list: list[NetworkValidatorException] = []
    gen_type._validate_emission_reduction(actual_exception_list, network)
    assert_same_exception_list(actual_exception_list, exception_list)


@pytest.mark.parametrize(
    "gen_type, exception_list",
    [
        (
            GeneratorType(
                **default_generator_type
                | {"name": "gen_type_no_conv_rate", "conversion_rate": None}
            ),
            [NetworkValidatorException("Conversion rate cannot be None.")],
        ),
        (
            GeneratorType(
                **default_generator_type
                | {
                    "name": "gen_type_non_existed_conv_rate",
                    "conversion_rate": {"DUMMY_CONV_RATE": 0.9},
                }
            ),
            [
                NetworkValidatorException(
                    "Conversion rate energy types do not exist "
                    "in network energy types: ['ELECTRICITY', 'HEATING']"
                )
            ],
        ),
    ],
)
def test_conversion_rate_validators(
    gen_type: GeneratorType,
    exception_list: list[NetworkValidatorException],
    network: Network,
) -> None:
    actual_exception_list: list[NetworkValidatorException] = []
    gen_type._validate_conversion_rate(actual_exception_list, network)
    assert_same_exception_list(actual_exception_list, exception_list)


@pytest.mark.parametrize(
    "params, exception_list",
    [
        pytest.param(
            {
                "power_utilization": pd.Series(
                    data=[1.0] * DEFAULT_HOURS,
                    index=np.arange(DEFAULT_HOURS),
                )
            },
            [],
            id="correct_power_utilization",
        ),
        pytest.param(
            {
                "power_utilization": 1.0,
            },
            [
                NetworkValidatorException(
                    "power_utilization must be a pandas Series, but float given"
                )
            ],
            id="incorrect_power_utilization_type",
        ),
        pytest.param(
            {
                "power_utilization": pd.Series(
                    data=[-1.0] * DEFAULT_HOURS,
                    index=np.arange(DEFAULT_HOURS),
                ),
            },
            [
                NetworkValidatorException(
                    "power_utilization values must be greater "
                    f"or equal 0, but for hours: {list(np.arange(DEFAULT_HOURS))} it is not"
                )
            ],
            id="incorrect_power_utilization_values",
        ),
        pytest.param(
            {
                "power_utilization": pd.Series(
                    data=[-1.0] * (DEFAULT_HOURS - 2) + [2.0, 2.5],
                    index=np.arange(DEFAULT_HOURS),
                ),
            },
            [
                NetworkValidatorException(
                    "power_utilization values must be greater "
                    f"or equal 0, but for hours: {list(np.arange(DEFAULT_HOURS-2))} it is not"
                )
            ],
            id="incorrect_mixed_power_utilization_values",
        ),
        pytest.param(
            {
                "minimal_power_utilization": pd.Series(
                    data=[0.2] * DEFAULT_HOURS,
                    index=np.arange(DEFAULT_HOURS),
                )
            },
            [],
            id="minimal_power_utilization_correct",
        ),
        pytest.param(
            {
                "minimal_power_utilization": 0.2,
            },
            [
                NetworkValidatorException(
                    "minimal_power_utilization must be a pandas Series, but float given"
                )
            ],
            id="minimal_power_utilization_float",
        ),
        pytest.param(
            {
                "minimal_power_utilization": pd.Series(
                    data=[-1.0] * DEFAULT_HOURS,
                    index=np.arange(DEFAULT_HOURS),
                ),
            },
            [
                NetworkValidatorException(
                    "minimal_power_utilization values must be greater "
                    f"or equal 0, but for hours: {list(np.arange(DEFAULT_HOURS))} it is not"
                )
            ],
            id="incorrect_minimal_power_utilization_values",
        ),
        pytest.param(
            {
                "minimal_power_utilization": pd.Series(
                    data=[0.8] * DEFAULT_HOURS,
                    index=np.arange(DEFAULT_HOURS),
                ),
                "power_utilization": pd.Series(
                    data=[0.4] * DEFAULT_HOURS,
                    index=np.arange(DEFAULT_HOURS),
                ),
            },
            [
                NetworkValidatorException(
                    "Power utilization values must be greater than minimal power utilization values, but for "
                    f"hours {list(np.arange(DEFAULT_HOURS))} they are not"
                )
            ],
            id="all_values_minimal_greater_than_normal",
        ),
        pytest.param(
            {
                "minimal_power_utilization": pd.Series(
                    data=[0.5] * DEFAULT_HOURS,
                    index=np.arange(DEFAULT_HOURS),
                ),
                "power_utilization": pd.Series(
                    data=[0.7] * (DEFAULT_HOURS - 2) + [0.3, 0.3],
                    index=np.arange(DEFAULT_HOURS),
                ),
            },
            [
                NetworkValidatorException(
                    "Power utilization values must be greater than minimal power utilization values, but for "
                    "hours [22, 23] they are not"
                )
            ],
            id="last_2_values_minimal_greater_than_normal",
        ),
    ],
)
def test_power_utilization(
    params: dict,
    exception_list: list[NetworkValidatorException],
    network: Network,
) -> None:
    gen_type = GeneratorType(**default_generator_type | params)
    actual_exception_list: list[NetworkValidatorException] = []
    gen_type._validate_power_utilization_boundaries(network, actual_exception_list)
    assert_same_exception_list(actual_exception_list, exception_list)


def test_efficiency_warning(network: Network) -> None:
    actual_exception_list: list[NetworkValidatorException] = []
    gen_type = GeneratorType(**default_generator_type)
    df = gen_type.efficiency
    df.loc[[10, 20], ["ELECTRICITY", "HEATING"]] = (0.6, 0.9)

    with pytest.warns(SumNotEqualToOneWarning) as warn:
        gen_type._validate_efficiency(actual_exception_list, network)

    assert (
        str(warn.list[0].message)
        == "Generator type default_generator_type efficiency contains hours: [10, 20] which sum for each energy "
        "type is above 1"
    )
    assert not actual_exception_list


@pytest.mark.parametrize(
    "compensation, exception_list",
    [
        pytest.param(
            pd.Series(
                [1, -0.3, 0, -5.9],
                index=np.arange(default_network_constants.n_years),
            ),
            [],
            id="happy_path",
        ),
        pytest.param(
            "compensation",
            [
                NetworkValidatorException(
                    "Generation compensation of generator type default "
                    "must be type of pandas Series or None."
                )
            ],
            id="wrong_data_type",
        ),
        pytest.param(
            pd.Series(
                ["t", -0.3, 0, -5.9],
                index=np.arange(default_network_constants.n_years),
            ),
            [
                NetworkValidatorException(
                    "Generation compensation of generator type default must "
                    "contain float or int values only."
                ),
            ],
            id="wrong_values_type_in_series",
        ),
    ],
)
def test_generation_compensation(
    compensation: Any,
    exception_list: list[NetworkValidatorException],
) -> None:
    actual_exception_list: list[NetworkValidatorException] = []
    gen_type = get_default_generator_type(series_length=DEFAULT_YEARS)
    gen_type.generation_compensation = compensation
    gen_type._validate_generation_compensation(actual_exception_list)
    assert_same_exception_list(actual_exception_list, exception_list)


@pytest.mark.parametrize(
    "ramp_down, ramp_up, exception_list",
    [
        pytest.param(0.2, 0.5, [], id="happy_path_both_float"),
        pytest.param(np.nan, np.nan, [], id="happy_path_both_nan"),
        pytest.param(
            "0.2",
            [0.3],
            [
                NetworkValidatorException("ramp_down value must be float or empty."),
                NetworkValidatorException("ramp_up value must be float or empty."),
            ],
            id="down_str_up_list",
        ),
        pytest.param(
            np.nan,
            1.05,
            [
                NetworkValidatorException(
                    "ramp_up value must be greater than 0 and less than 1, but it is 1.05"
                )
            ],
            id="up_above_value",
        ),
    ],
)
def test_generator_type_validation_ramp(
    ramp_up: Any,
    ramp_down: Any,
    exception_list: list[NetworkValidatorException],
) -> None:
    actual_exception_list: list[NetworkValidatorException] = []
    gen_type = get_default_generator_type(
        series_length=default_network_constants.n_years,
        ramp_down=ramp_down,
        ramp_up=ramp_up,
    )
    gen_type._validate_ramp(actual_exception_list)
    assert_same_exception_list(actual_exception_list, exception_list)
