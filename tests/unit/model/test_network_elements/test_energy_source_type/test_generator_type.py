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

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.exceptions import NetworkValidatorException
from pyzefir.model.network import Network
from pyzefir.model.network_elements import GeneratorType
from pyzefir.model.network_elements.bus import Bus
from pyzefir.model.network_elements.energy_sources.generator import Generator
from pyzefir.model.network_elements.fuel import Fuel
from tests.unit.defaults import (
    DEFAULT_HOURS,
    ELECTRICITY,
    HEATING,
    default_generator_type,
    default_network_constants,
    get_default_generator_type,
)
from tests.unit.model.test_network_elements.helpers import assert_same_exception_list


@pytest.fixture
def network() -> Network:
    network = Network(
        energy_types=[ELECTRICITY, HEATING],
        network_constants=default_network_constants,
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


@pytest.mark.parametrize(
    "gen_type, exception_list",
    [
        (
            get_default_generator_type(
                series_length=default_network_constants.n_years, ramp="sadasd"
            ),
            [NetworkValidatorException("Ramp value must be float or empty.")],
        ),
        (
            get_default_generator_type(
                series_length=default_network_constants.n_years,
                name="gen_type_cap_fuel",
                capacity_factor="dummy_cap_f",
            ),
            [
                NetworkValidatorException(
                    "Generator type gen_type_cap_fuel can have either capacity factor or fuel at the same time"
                ),
                NetworkValidatorException(
                    "Generator type 'gen_type_cap_fuel' capacity factor 'dummy_cap_f' has not been added to the network"
                ),
            ],
        ),
        (
            get_default_generator_type(
                series_length=default_network_constants.n_years,
                name="gen_type_cap_fuel",
                ramp=1,
            ),
            [
                NetworkValidatorException(
                    "Ramp value for gen_type_cap_fuel must be greater than 0 and less than 1, but it is 1"
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
                    "None or str type for fuel expected but type: <class 'int'> for generator type: gen_type_E given"
                ),
                NetworkValidatorException(
                    "Generator gen_type_E fuel 1 has not been added to the network"
                ),
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
                    "None or str type for capacity factor expected but type: <class 'int'> for generator type: "
                    "gen_type_F given"
                ),
                NetworkValidatorException(
                    "Generator type 'gen_type_F' capacity factor '1' has not been added to the network"
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
                    "Generator type 'gen_type_I' capacity factor 'test_gen_I' has not been added to the network"
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
        (
            GeneratorType(
                **default_generator_type
                | {
                    "name": "gen_type_L",
                    "efficiency": {"DUMMY_ELECTRICITY": 0.9, "DUMMY_HEATING": 0.6},
                }
            ),
            [
                NetworkValidatorException(
                    "Efficiency energy types of gen_type_L do not exist in network energy types: "
                    "['ELECTRICITY', 'HEATING']"
                )
            ],
        ),
        (
            GeneratorType(
                **default_generator_type
                | {"name": "gen_type_no_eff", "efficiency": None}
            ),
            [
                NetworkValidatorException(
                    "Efficiency of generator type: gen_type_no_eff cannot be None."
                )
            ],
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
        (
            GeneratorType(
                **default_generator_type
                | {"name": "gen_type_no_e_r", "emission_reduction": None}
            ),
            [
                NetworkValidatorException(
                    "Emission reduction of generator type: gen_type_no_e_r cannot be None."
                )
            ],
        ),
        (
            GeneratorType(
                **default_generator_type
                | {
                    "name": "gen_type_non_existed_e_r",
                    "emission_reduction": {"DUMMY_E_R": 0.9},
                }
            ),
            [
                NetworkValidatorException(
                    "Emission reduction emission types {'DUMMY_E_R'} of "
                    "gen_type_non_existed_e_r do not exist in network "
                    "emission types: ['CO2', 'PM10']"
                )
            ],
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
            [
                NetworkValidatorException(
                    "Conversion rate of generator type: gen_type_no_conv_rate cannot be None."
                )
            ],
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
                    "Conversion rate energy types of gen_type_non_existed_conv_rate do not exist "
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
                    "Power utilization values for default_generator_type must be greater "
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
                    "Power utilization values for default_generator_type must be greater "
                    f"or equal 0, but for hours: {list(np.arange(DEFAULT_HOURS-2))} it is not"
                )
            ],
            id="incorrect_mixed_power_utilization_values",
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
    gen_type._validate_power_utilization(network, actual_exception_list)
    assert_same_exception_list(actual_exception_list, exception_list)
