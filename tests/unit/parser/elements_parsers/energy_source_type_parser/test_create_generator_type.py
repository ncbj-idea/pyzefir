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

from typing import Literal

import numpy as np
import pandas as pd
import pytest

from pyzefir.parser.elements_parsers.energy_source_type_parser import (
    EnergySourceTypeParser,
    EnergySourceTypeParserException,
)
from tests.unit.defaults import DEFAULT_HOURS, ELECTRICITY, HEATING
from tests.unit.parser.elements_parsers.utils import assert_equal


@pytest.mark.parametrize(
    ("name", "expected_results"),
    [
        pytest.param(
            "GEN_TYPE_3",
            {
                "name": "GEN_TYPE_3",
                "life_time": 15,
                "build_time": 0,
                "capex": pd.Series([90, 88, 85]),
                "opex": pd.Series([15, 14, 12]),
                "min_capacity": pd.Series([np.nan] * 4),
                "max_capacity": pd.Series([np.nan] * 4),
                "min_capacity_increase": pd.Series([np.nan] * 4),
                "max_capacity_increase": pd.Series([np.nan] * 4),
                "efficiency": pd.DataFrame({HEATING: [0.84] * 8760}),
                "energy_types": {HEATING},
                "emission_reduction": {
                    "CO2": pd.Series([0.4, 0.5, 0.5, 0.6]),
                    "SO2": pd.Series([0.25, 0.25, 0.25, 0.25]),
                },
                "conversion_rate": {ELECTRICITY: pd.Series(np.ones(8760))},
                "fuel": None,
                "capacity_factor": None,
                "power_utilization": pd.Series(data=[0.9] * 24, index=np.arange(24)),
                "minimal_power_utilization": pd.Series(
                    data=[0.2] * 24, index=np.arange(24)
                ),
            },
            id="no_optional_parameters_provided",
        ),
    ],
)
def test_create_generator_type(
    name: str,
    expected_results: dict,
    energy_source_type_parser: EnergySourceTypeParser,
) -> None:
    df_row = pd.Series(
        index=[
            "name",
            "build_time",
            "life_time",
            "power_utilization",
            "minimal_power_utilization",
        ],
        data=[name, 0, 15, 0.9, 0.2],
    )
    energy_source_type_df = (
        energy_source_type_parser._prepare_energy_source_parameters()
    )
    efficiency_df = energy_source_type_parser.generators_efficiency.pivot_table(
        index=energy_source_type_parser.generators_efficiency.index,
        columns="energy_type",
        values="efficiency",
    )
    demand_dict = energy_source_type_parser._prepare_conversion_rate_dict(
        energy_source_type_parser.conversion_rate
    )
    emission_reduction_dict = (
        energy_source_type_parser._prepare_generator_emission_reduction(
            energy_source_type_parser.generators_emission_reduction,
            energy_source_type_parser.yearly_emission_reduction,
            energy_source_type_parser.n_years,
        )
    )
    result = energy_source_type_parser._create_generator_type(
        df_row,
        energy_source_type_df,
        efficiency_df,
        demand_dict,
        emission_reduction_dict,
    )

    for attr, value in expected_results.items():
        to_compare = getattr(result, attr)
        assert_equal(value, to_compare)


@pytest.mark.parametrize(
    "params, utilization_type, expected_results",
    [
        pytest.param(
            {
                "name": "GEN_TYPE_3",
                "power_utilization": 1.0,
                "minimal_power_utilization": 0.2,
            },
            "power_utilization",
            pd.Series(data=[1.0] * DEFAULT_HOURS, index=np.arange(DEFAULT_HOURS)),
            id="float_value_passed",
        ),
        pytest.param(
            {
                "name": "GEN_TYPE_1",
                "power_utilization": np.nan,
                "minimal_power_utilization": 0.2,
            },
            "power_utilization",
            pd.Series(data=[0.9] * 8760, index=np.arange(8760)),
            id="value_is_nan",
        ),
        pytest.param(
            {
                "name": "test",
                "power_utilization": np.nan,
                "minimal_power_utilization": 0.2,
            },
            "power_utilization",
            pd.Series(data=[1.0] * DEFAULT_HOURS, index=np.arange(DEFAULT_HOURS)),
            id="value_is_nan_and_no_gen_in_df",
        ),
        pytest.param(
            {
                "name": "GEN_TYPE_3",
                "power_utilization": 1.0,
                "minimal_power_utilization": 0.3,
            },
            "minimal_power_utilization",
            pd.Series(data=[0.3] * DEFAULT_HOURS, index=np.arange(DEFAULT_HOURS)),
            id="float_value_passed_minimal",
        ),
        pytest.param(
            {
                "name": "GEN_TYPE_1",
                "power_utilization": 1.0,
                "minimal_power_utilization": np.nan,
            },
            "minimal_power_utilization",
            pd.Series(data=[0.2] * 8760, index=np.arange(8760)),
            id="value_is_nan_minimal",
        ),
    ],
)
def test_get_power_utilization(
    params: dict,
    expected_results: float | pd.Series | list[EnergySourceTypeParserException],
    energy_source_type_parser: EnergySourceTypeParser,
    utilization_type: Literal["power_utilization", "minimal_power_utilization"],
) -> None:
    power_utilization_df = (
        energy_source_type_parser.generators_power_utilization
        if utilization_type == "power_utilization"
        else energy_source_type_parser.generators_minimal_power_utilization
    )
    df_row = pd.Series({"build_time": 0, "life_time": 15} | params)
    result = energy_source_type_parser._get_power_utilization_boundaries(
        df_row["name"], df_row, power_utilization_df, utilization_type
    )
    pd.testing.assert_series_equal(
        result, expected_results, check_names=False, check_index_type=False
    )


def test_get_power_utilization_two_sources(
    energy_source_type_parser: EnergySourceTypeParser,
) -> None:
    df_row = pd.Series(
        {
            "build_time": 0,
            "life_time": 15,
            "name": "GEN_TYPE_1",
            "power_utilization": 1.0,
        }
    )
    with pytest.raises(EnergySourceTypeParserException) as e_info:
        energy_source_type_parser._get_power_utilization_boundaries(
            df_row["name"],
            df_row,
            energy_source_type_parser.generators_power_utilization,
            "power_utilization",
        )
    assert str(e_info.value) == (
        "power_utilization for GEN_TYPE_1 must be specified by passing value "
        "in generator_types.xlsx: Generator Types sheet "
        "or Power Utilization sheet, "
        "but two methods were used at once"
    )


@pytest.mark.parametrize(
    "df_er, df_yer, n_years, expected",
    [
        pytest.param(
            pd.DataFrame(
                {"CO2": [0.1, 0.1], "SO2": [0.1, 0.1]},
                index=["Generator1", "Generator2"],
            ),
            pd.DataFrame(
                {
                    "emission_type": ["CO2", "CO2", "SO2", "SO2"],
                    "year": [1, 2, 1, 2],
                    "Generator1": [0.2, 0.2, 0.3, 0.3],
                    "Generator2": [0.5, 0.5, 0.4, 0.3],
                }
            ),
            3,
            {
                "Generator1": {
                    "CO2": pd.Series([0.1, 0.2, 0.2]),
                    "SO2": pd.Series([0.1, 0.3, 0.3]),
                },
                "Generator2": {
                    "CO2": pd.Series([0.1, 0.5, 0.5]),
                    "SO2": pd.Series([0.1, 0.4, 0.3]),
                },
            },
            id="3 years both emission many values ",
        ),
        pytest.param(
            pd.DataFrame({"CO2": [0.2], "SO2": [0.2]}, index=["Generator3"]),
            pd.DataFrame(
                {
                    "emission_type": ["CO2", "SO2"],
                    "year": [1, 1],
                    "Generator3": [0.3, 0.4],
                }
            ),
            2,
            {
                "Generator3": {
                    "CO2": pd.Series([0.2, 0.3]),
                    "SO2": pd.Series([0.2, 0.4]),
                }
            },
            id="2 years both emissions single values",
        ),
        pytest.param(
            pd.DataFrame(
                {"CO2": [0.1, 0.1], "SO2": [0.1, 0.1]},
                index=["Generator1", "Generator2"],
            ),
            pd.DataFrame(
                {
                    "emission_type": ["CO2", "CO2", "SO2", "SO2"],
                    "year": [1, 2, 1, 2],
                    "Generator1": [None, None, 0.2, 0.2],
                    "Generator2": [0.5, 0.5, 0.5, 0.5],
                }
            ),
            3,
            {
                "Generator1": {
                    "CO2": pd.Series([0.1, 0.1, 0.1]),
                    "SO2": pd.Series([0.1, 0.2, 0.2]),
                },
                "Generator2": {
                    "CO2": pd.Series([0.1, 0.5, 0.5]),
                    "SO2": pd.Series([0.1, 0.5, 0.5]),
                },
            },
            id="3 years Gen1 no CO2 emissions ",
        ),
        pytest.param(
            pd.DataFrame(
                {"CO2": [0.1, 0.1], "SO2": [0.1, 0.1]},
                index=["Generator1", "Generator2"],
            ),
            pd.DataFrame(),
            5,
            {
                "Generator1": {
                    "CO2": pd.Series([0.1, 0.1, 0.1, 0.1, 0.1]),
                    "SO2": pd.Series([0.1, 0.1, 0.1, 0.1, 0.1]),
                },
                "Generator2": {
                    "CO2": pd.Series([0.1, 0.1, 0.1, 0.1, 0.1]),
                    "SO2": pd.Series([0.1, 0.1, 0.1, 0.1, 0.1]),
                },
            },
            id="5 years yearly emissions empty dataframe ",
        ),
    ],
)
def test_prepare_generator_emission_reduction(
    energy_source_type_parser: EnergySourceTypeParser,
    df_er: pd.DataFrame,
    df_yer: pd.DataFrame,
    n_years: int,
    expected: dict[str, dict[str, pd.Series]],
) -> None:
    result = energy_source_type_parser._prepare_generator_emission_reduction(
        df_er, df_yer, n_years
    )
    for generator in expected:
        for emission_type in expected[generator]:
            pd.testing.assert_series_equal(
                result[generator][emission_type], expected[generator][emission_type]
            )
