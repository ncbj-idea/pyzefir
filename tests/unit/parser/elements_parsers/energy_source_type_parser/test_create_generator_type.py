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
                "efficiency": {HEATING: 0.84},
                "energy_types": {HEATING},
                "emission_reduction": {"CO2": 0.4, "SO2": 0.25},
                "conversion_rate": {ELECTRICITY: pd.Series(np.ones(8760))},
                "fuel": None,
                "capacity_factor": None,
                "power_utilization": pd.Series(data=[0.9] * 24, index=np.arange(24)),
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
        index=["name", "build_time", "life_time", "power_utilization"],
        data=[name, 0, 15, 0.9],
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
    result = energy_source_type_parser._create_generator_type(
        df_row, energy_source_type_df, efficiency_df, demand_dict
    )

    for attr, value in expected_results.items():
        to_compare = getattr(result, attr)
        assert_equal(value, to_compare)


@pytest.mark.parametrize(
    "params, expected_results",
    [
        pytest.param(
            {"name": "GEN_TYPE_3", "power_utilization": 1.0},
            pd.Series(data=[1.0] * DEFAULT_HOURS, index=np.arange(DEFAULT_HOURS)),
            id="float_value_passed",
        ),
        pytest.param(
            {"name": "GEN_TYPE_1", "power_utilization": np.nan},
            pd.Series(data=[0.9] * 8760, index=np.arange(8760)),
            id="value_is_nan",
        ),
        pytest.param(
            {"name": "test", "power_utilization": np.nan},
            pd.Series(data=[1.0] * DEFAULT_HOURS, index=np.arange(DEFAULT_HOURS)),
            id="value_is_nan_and_no_gen_in_df",
        ),
    ],
)
def test_get_power_utilization(
    params: dict,
    expected_results: float | pd.Series | list[EnergySourceTypeParserException],
    energy_source_type_parser: EnergySourceTypeParser,
) -> None:
    df_row = pd.Series({"build_time": 0, "life_time": 15} | params)
    result = energy_source_type_parser._get_power_utilization(df_row["name"], df_row)
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
        energy_source_type_parser._get_power_utilization(df_row["name"], df_row)
    assert str(e_info.value) == (
        "Power utilization for GEN_TYPE_1 must be specified by passing value "
        "in generator_types.xlsx: Generator Types sheet "
        "or Power Utilization sheet, "
        "but two methods were used at once"
    )
