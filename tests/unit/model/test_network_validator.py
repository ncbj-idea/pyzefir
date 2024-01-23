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

from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network import Network
from pyzefir.model.network_validator import (
    BaseCapacityValidator,
    BaseTotalEmissionValidation,
    NetworkElementsValidation,
    NetworkValidator,
    RelativeEmissionLimitsValidation,
)
from pyzefir.model.utils import NetworkConstants
from pyzefir.parser.csv_parser import CsvParser
from pyzefir.parser.network_creator import NetworkCreator
from pyzefir.utils.path_manager import CsvPathManager
from tests.unit.defaults import CO2_EMISSION, ELECTRICITY, HEATING, PM10_EMISSION
from tests.unit.model.test_network_elements.helpers import assert_same_exception_list
from tests.unit.optimization.gurobi.test_model.utils import (
    set_network_elements_parameters,
)


@pytest.fixture
def network(csv_root_path: Path) -> Network:
    loaded_csv_data = CsvParser(
        path_manager=CsvPathManager(
            dir_path=csv_root_path,
            scenario_name="scenario_1",
        )
    ).load_dfs()
    return NetworkCreator.create(loaded_csv_data)


@pytest.mark.parametrize(
    "relative_emission_limits, exception_msg",
    (
        pytest.param(
            {
                CO2_EMISSION: [1, 2, 3, 4],
                PM10_EMISSION: pd.Series([0.95, 0.85, 0.75], index=range(1, 4)),
            },
            "Relative emission limits must be type of dict[str, pd.Series].",
            id="Relative emission limits invalid type",
        ),
        pytest.param(
            {
                "test": pd.Series([0.95, 0.85, 0.75], index=range(1, 4)),
                PM10_EMISSION: pd.Series([0.95, 0.85, 0.75], index=range(1, 4)),
            },
            (
                "Emission types in relative emission limits must correspond to "
                "the names of the emission types listed in structure / Emission Types."
            ),
            id="Network emission types not in rel em lim keys",
        ),
        pytest.param(
            {
                CO2_EMISSION: pd.Series([1.95, 0.85, 10.75], index=range(1, 4)),
                PM10_EMISSION: pd.Series([0.95, 0.85, 0.75], index=range(1, 4)),
            },
            (
                "In each column corresponding to the emission type, "
                "the expected value must be in the range [0,1], "
                "but for CO2 it's [ 1.95  0.85 10.75]"
            ),
            id="Relative emission limits invalid values",
        ),
        pytest.param(
            {
                CO2_EMISSION: pd.Series(["1.25", 0.85, 0.75], index=range(1, 4)),
                PM10_EMISSION: pd.Series([0.95, 0.85, "test"], index=range(1, 4)),
            },
            (
                "In each column corresponding to the emission type, "
                "the expected value type is float or int, "
                "but for CO2 it's [<class 'str'>, <class 'float'>, <class 'float'>]."
            ),
            id="Relative emission limits invalid values type",
        ),
        pytest.param(
            {
                CO2_EMISSION: pd.Series([0.95, 0.85, 0.75], index=range(0, 3)),
                PM10_EMISSION: pd.Series([0.95, 0.85, 0.75], index=range(1, 4)),
            },
            (
                "Year indices must start with 1 - you cannot specify "
                "a base year index with 0 in the column, "
                "but for CO2 base index exists in input data."
            ),
            id="Relative emission limits defined for based year",
        ),
    ),
)
def test_relative_emission_limits_validation(
    relative_emission_limits: dict[str, pd.Series],
    exception_msg: str,
) -> None:
    base_total_emission = {CO2_EMISSION: np.nan, PM10_EMISSION: np.nan}
    network = Network(
        network_constants=NetworkConstants(
            4, 24, relative_emission_limits, base_total_emission
        ),
        energy_types=[ELECTRICITY, HEATING],
        emission_types=[CO2_EMISSION, PM10_EMISSION],
    )
    exception_list: list[NetworkValidatorException] = []
    RelativeEmissionLimitsValidation.validate(network, exception_list=exception_list)
    assert len(exception_list)
    assert str(exception_list[0]) == exception_msg


@pytest.mark.parametrize(
    "base_total_emission",
    (
        pytest.param(
            {
                1: 1000,
                PM10_EMISSION: 2000,
            },
            id="Base total emission key is not str",
        ),
        pytest.param(
            {
                CO2_EMISSION: [0.95, 0.85, 0.75],
                PM10_EMISSION: "abc",
            },
            id="Base total emission value is not float/int",
        ),
        pytest.param(
            [
                1000,
                2000,
            ],
            id="Base total emission is not dict but list",
        ),
    ),
)
def test_base_total_emission_validation(
    base_total_emission: dict[str, float | int],
) -> None:
    relative_emission_limits = {
        CO2_EMISSION: pd.Series([np.nan] * 4),
        PM10_EMISSION: pd.Series([np.nan] * 4),
    }

    network = Network(
        network_constants=NetworkConstants(
            4, 24, relative_emission_limits, base_total_emission
        ),
        energy_types=[ELECTRICITY, HEATING],
        emission_types=[CO2_EMISSION, PM10_EMISSION],
    )
    exception_list: list[NetworkValidatorException] = []
    BaseTotalEmissionValidation.validate(network, exception_list)
    assert len(exception_list)
    assert (
        str(exception_list[0])
        == "Base total emission should be type of dict[str, float | int]."
    )


@pytest.mark.parametrize(
    "element_name, element_params, exception_msg",
    (
        pytest.param(
            "buses",
            {"HS1": {"name": ["'HS1'"]}},
            "Network element name must be of type string, but it is <class 'list'> instead.",
            id="Renaming bus to the wrong name type",
        ),
        pytest.param(
            "generator_types",
            {"CHP_COAL": {"fuel": "uranium"}},
            "Generator CHP_COAL fuel uranium has not been added to the network",
            id="Changing the generator_type CHP_COAL fuel that is not available in the network",
        ),
        pytest.param(
            "local_balancing_stacks",
            {"LKT2": {"buses_out": ["LKT2_EE", "LKT2_H"]}},
            "Outlet buses of local balancing stack LKT2 must be a dict, not <class 'list'>.",
            id="Changing type buses_out to list instead of required dict",
        ),
        pytest.param(
            "capacity_factors",
            {"SUN": {"profile": pd.Series([4] * 12)}},
            "Profile must have 24 values",
            id="Changing SUN capacity factor series length to 12 instead of required 24 ",
        ),
    ),
)
def test_network_elements_validation(
    network: Network,
    element_name: str,
    element_params: dict,
    exception_msg: str,
) -> None:
    set_network_elements_parameters(getattr(network, element_name), element_params)
    exception_list: list[NetworkValidatorExceptionGroup] = []

    NetworkElementsValidation.validate(network, exception_list)
    assert len(exception_list)
    assert str(exception_list[0].exceptions[0]) == exception_msg


def test_network_validator(network: Network) -> None:
    with mock.patch.object(
        RelativeEmissionLimitsValidation, "validate"
    ) as mock_emission_validate, mock.patch.object(
        NetworkElementsValidation, "validate"
    ) as mock_element_validate:
        NetworkValidator(network).validate()

        mock_emission_validate.assert_called_once_with(network, [])
        mock_element_validate.assert_called_once_with(network, [])


def test_network_relative_emission_limit_happy_path(network: Network) -> None:
    exception_list: list[NetworkValidatorException] = []
    RelativeEmissionLimitsValidation.validate(network, exception_list=exception_list)

    assert not exception_list


def test_network_elements_validation_happy_path(network: Network) -> None:
    exception_list: list[NetworkValidatorException] = []
    NetworkElementsValidation.validate(network, exception_list=exception_list)

    assert not exception_list


def test_network_validator_happy_path(network: Network) -> None:
    NetworkValidator(network).validate()


def test_base_capacity_validator(network: Network) -> None:
    exception_list: list[NetworkValidatorException] = []
    BaseCapacityValidator.validate(network, exception_list=exception_list)

    assert not exception_list


def test_surplus_lbs_per_aggr(network: Network) -> None:
    exception_list: list[NetworkValidatorException] = []
    network.aggregated_consumers["DOMKI"].stack_base_fraction["LKT2"] = 0
    BaseCapacityValidator.validate(network, exception_list=exception_list)

    assert len(exception_list) == 1
    assert (
        str(exception_list[0])
        == "Stack LKT2 is connected to more than one aggregated consumer"
    )


def test_unit_not_connected_to_single_stack(network: Network) -> None:
    exception_list: list[NetworkValidatorException] = []
    network.local_balancing_stacks["LKT3"].buses["ELECTRICITY"].add("KSE")
    BaseCapacityValidator.validate(network, exception_list=exception_list)

    assert len(exception_list) > 1
    assert (
        str(exception_list[0])
        == "Each generator (CHP_BIOMASS_HS1_KSE) must be used exactly in one or zero stacks."
    )


@pytest.mark.parametrize(
    "unit_name, prop, value, exception_list",
    (
        (
            "BOILER_COAL_LKT2",
            "unit_base_cap",
            None,
            [
                NetworkValidatorException(
                    "Base capacity for unit BOILER_COAL_LKT2 is not defined"
                )
            ],
        ),
        (
            "BOILER_COAL_LKT2",
            "min_device_nom_power",
            None,
            [
                NetworkValidatorException(
                    "For units (BOILER_COAL_LKT2) that are used in a local balancing stack, "
                    "attributes min_device_power and max_device_power must be defined"
                )
            ],
        ),
        (
            "BOILER_COAL_LKT2",
            "unit_base_cap",
            0,
            [
                NetworkValidatorException(
                    "In energy source BOILER_COAL_LKT2, if base capacity has been defined, "
                    "the compound inequality must be true: base_fraction * n_consumers "
                    "* min_device_nom_power <= base_capacity <= base_fraction * n_consumers "
                    "* max_device_nom_power, but it is "
                    "0.7 * 50000 * 10.0 (350000.0) <= 0 <= 0.7 * 50000 * 30.0 (1050000.0) instead"
                )
            ],
        ),
        (
            "HEAT_PLANT_COAL_HS2",
            "max_device_nom_power",
            0,
            [
                NetworkValidatorException(
                    "For units (HEAT_PLANT_COAL_HS2) that are not used in a local balancing stack, "
                    "attributes min_device_power and max_device_power must not be defined"
                )
            ],
        ),
    ),
)
def test_base_capacity_values(
    unit_name: str,
    prop: str,
    value: Any,
    exception_list: list[Exception],
    network: Network,
) -> None:
    actual_exception_list: list[NetworkValidatorException] = []
    setattr(network.generators[unit_name], prop, value)
    BaseCapacityValidator.validate(network, exception_list=actual_exception_list)

    assert_same_exception_list(actual_exception_list, exception_list)
