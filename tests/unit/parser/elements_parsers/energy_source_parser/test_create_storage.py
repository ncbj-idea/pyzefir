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

from pyzefir.model.network_elements import Storage
from pyzefir.parser.elements_parsers.energy_source_unit_parser import (
    EnergySourceUnitParser,
)


def test_create_storage_no_optional_params(
    energy_source_unit_parser: EnergySourceUnitParser,
) -> None:
    """Test if storage will be created correctly with no optional parameters provided."""
    stor = energy_source_unit_parser._create_storage(
        energy_source_unit_parser.df_storages.iloc[1, :]
    )

    assert isinstance(stor, Storage)
    assert stor.name == "STORAGE_2"
    assert stor.energy_source_type == "HEAT_STORAGE"
    assert stor.bus == "HAS"
    assert stor.unit_base_cap == 9000
    assert pd.Series([None] * 4).equals(stor.unit_max_capacity)
    assert pd.Series([None] * 4).equals(stor.unit_min_capacity)
    assert pd.Series([None] * 4).equals(stor.unit_max_capacity_increase)
    assert pd.Series([None] * 4).equals(stor.unit_min_capacity_increase)


@pytest.mark.parametrize(
    ("storage_name", "expected_params"),
    [
        pytest.param(
            "STORAGE_1",
            {
                "unit_min_capacity": pd.Series([2.0, 2.0, 2.0, 2.0]),
                "unit_max_capacity": pd.Series([np.nan] * 4),
                "unit_min_capacity_increase": pd.Series([np.nan] * 4),
                "unit_max_capacity_increase": pd.Series([np.nan] * 4),
            },
            id="unit_max_cap_only",
        ),
        pytest.param(
            "STORAGE_2",
            {
                "unit_min_capacity": pd.Series([1.0, 1.0, 1.0, 1.0]),
                "unit_max_capacity": pd.Series([1.0, 1.0, 1.0, 1.0]),
                "unit_min_capacity_increase": pd.Series([np.nan] * 4),
                "unit_max_capacity_increase": pd.Series([np.nan] * 4),
            },
            id="unit_cap_min_and_unit_cap_max_only",
        ),
        pytest.param(
            "STORAGE_3",
            {
                "unit_min_capacity": pd.Series([np.nan] * 4),
                "unit_max_capacity": pd.Series([np.nan] * 4),
                "unit_min_capacity_increase": pd.Series([np.nan] * 4),
                "unit_max_capacity_increase": pd.Series([np.nan] * 4),
            },
            id="all_empty",
        ),
    ],
)
def test_create_generator_with_optional_parameters(
    storage_name: str,
    expected_params: dict[str, pd.Series],
    energy_source_unit_parser: EnergySourceUnitParser,
    technology_evolution_mock_df: pd.DataFrame,
) -> None:
    """Test if generator will be created correctly for varius configuration of given / non given optional params."""
    energy_source_unit_parser.df_element_energy_evolution = technology_evolution_mock_df
    stor_df = energy_source_unit_parser.df_storages
    stor = energy_source_unit_parser._create_storage(
        stor_df[stor_df["name"] == storage_name].squeeze()
    )

    assert isinstance(stor, Storage)
    assert stor.name == storage_name
    for param_name, expected_value in expected_params.items():
        assert expected_value.equals(getattr(stor, param_name))
