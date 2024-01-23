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
)
from tests.unit.defaults import ELECTRICITY, HEATING


@pytest.mark.parametrize(
    ("conv_rate", "expected_result"),
    [
        pytest.param(
            {
                "GEN_TYPE_2": pd.DataFrame(
                    columns=["hour_idx", ELECTRICITY],
                    data=[
                        [0, 1],
                        [1, 1],
                        [2, 2],
                    ],
                )
            },
            {"GEN_TYPE_2": {ELECTRICITY: pd.Series([1, 1, 2])}},
            id="one_energy_source_type_with_one_energy_type_conversion_rate",
        ),
        pytest.param(
            {
                "GEN_TYPE_1": pd.DataFrame(
                    columns=["hour_idx", ELECTRICITY, HEATING],
                    data=[
                        [0, 1, 0],
                        [1, 1, 10],
                        [2, 2, 3],
                    ],
                ),
                "GEN_TYPE_2": pd.DataFrame(
                    columns=["hour_idx", ELECTRICITY],
                    data=[
                        [0, 1],
                        [1, 1],
                        [2, 2],
                    ],
                ),
            },
            {
                "GEN_TYPE_1": {
                    ELECTRICITY: pd.Series([1, 1, 2]),
                    HEATING: pd.Series([0, 10, 3]),
                },
                "GEN_TYPE_2": {ELECTRICITY: pd.Series([1, 1, 2])},
            },
            id="two_gen_types_two_energy_types",
        ),
        pytest.param({}, {}, id="empty_input"),
    ],
)
def test_prepare_conversion_rate_dict(
    conv_rate: dict[str, pd.DataFrame],
    expected_result: dict[str, dict[str, pd.Series]],
) -> None:
    result = EnergySourceTypeParser._prepare_conversion_rate_dict(
        conversion_rate=conv_rate
    )

    assert set(result) == set(expected_result)
    for key in result:
        result_dict, expected_dict = result[key], expected_result[key]
        assert set(result_dict) == set(expected_dict)
        assert all(all(result_dict[k] == expected_dict[k]) for k in result_dict)


@pytest.mark.parametrize(
    ("cost_data", "capacity_evolution_data", "expected_result"),
    [
        pytest.param(
            pd.DataFrame(
                columns=["year_idx", "technology_type", "CAPEX", "OPEX"],
                data=[
                    [0, "GEN_TYPE_1", 10, 10],
                    [1, "GEN_TYPE_1", 8, 5],
                    [0, "BLABLA", 2, 1],
                    [1, "BLABLA", 25, 100],
                ],
            ),
            pd.DataFrame(
                columns=[
                    "year_idx",
                    "technology_type",
                    "max_capacity",
                    "min_capacity",
                    "max_capacity_increase",
                    "min_capacity_increase",
                ],
                data=[
                    [0, "GEN_TYPE_1", 1, 1, 1, 1],
                    [1, "GEN_TYPE_1", 0, 0, 0, 0],
                    [0, "BLABLA", np.nan, 1, 1, np.nan],
                    [1, "BLABLA", np.nan, 1, 1, np.nan],
                ],
            ),
            {
                "GEN_TYPE_1": pd.DataFrame.from_dict(
                    {
                        "CAPEX": [10, 8],
                        "OPEX": [8, 5],
                        "max_capacity": [1, 0],
                        "min_capacity": [1, 0],
                        "max_capacity_increase": [1, 0],
                        "min_capacity_increase": [1, 0],
                    }
                ),
                "BLABLA": pd.DataFrame.from_dict(
                    {
                        "CAPEX": [2, 25],
                        "OPEX": [1, 100],
                        "max_capacity": [np.nan, np.nan],
                        "min_capacity": [1, 1],
                        "max_capacity_increase": [1, 1],
                        "min_capacity_increase": [np.nan, np.nan],
                    }
                ),
            },
            id="simple_example",
        ),
        pytest.param(
            pd.DataFrame(columns=["year_idx", "technology_type", "CAPEX", "OPEX"]),
            pd.DataFrame(
                columns=[
                    "year_idx",
                    "technology_type",
                    "max_capacity",
                    "min_capacity",
                    "max_capacity_increase",
                    "min_capacity_increase",
                ]
            ),
            {},
            id="empty_input_data",
        ),
    ],
)
def test_prepare_energy_source_parameters_cost_parameters(
    energy_source_type_parser: EnergySourceTypeParser,
    cost_data: pd.DataFrame,
    capacity_evolution_data: pd.DataFrame,
    expected_result: pd.DataFrame,
) -> None:
    energy_source_type_parser.cost_parameters_df = cost_data
    energy_source_type_parser.energy_mix_evolution_limits_df = capacity_evolution_data
    result = energy_source_type_parser._prepare_energy_source_parameters()
    assert set(result) == set(expected_result)
    assert all(all(result[k] == expected_result[k]) for k in result)
