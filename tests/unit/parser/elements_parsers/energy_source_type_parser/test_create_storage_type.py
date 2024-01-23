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
from tests.unit.defaults import HEATING
from tests.unit.parser.elements_parsers.utils import assert_equal


@pytest.mark.parametrize(
    ("name", "expected_results"),
    [
        pytest.param(
            "STORAGE_TYPE_3",
            {
                "name": "STORAGE_TYPE_3",
                "life_time": 15,
                "build_time": 1,
                "capex": pd.Series([200, 150, 120]),
                "opex": pd.Series([10, 5, 3]),
                "min_capacity": pd.Series([np.nan] * 4),
                "max_capacity": pd.Series([np.nan] * 4),
                "min_capacity_increase": pd.Series([np.nan] * 4),
                "max_capacity_increase": pd.Series([np.nan] * 4),
                "energy_type": HEATING,
                "generation_efficiency": 0.88,
                "load_efficiency": 0.82,
                "cycle_length": 2190,
                "power_to_capacity": 10.0,
            },
            id="no_optional_parameters_provided",
        ),
    ],
)
def test_create_storage_type(
    name: str, expected_results: dict, energy_source_type_parser: EnergySourceTypeParser
) -> None:
    energy_source_type_df = (
        energy_source_type_parser._prepare_energy_source_parameters()
    )
    row_df = energy_source_type_parser.storage_type_df.loc[name, :]
    result = energy_source_type_parser._create_storage_type(
        row_df, energy_source_type_df
    )

    for attr, value in expected_results.items():
        to_compare = getattr(result, attr)
        assert_equal(value, to_compare)
