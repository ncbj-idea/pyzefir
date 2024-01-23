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

import pandas as pd
import pytest

from pyzefir.model.network_elements.emission_fee import EmissionFee
from pyzefir.parser.elements_parsers.emission_fee_parser import EmissionFeeParser
from pyzefir.utils.path_manager import DataCategories, DataSubCategories


@pytest.fixture
def emission_fee_dfs(csv_root_path: Path) -> list[pd.DataFrame]:
    emission_type_fee_df = pd.read_csv(
        csv_root_path
        / f"{DataCategories.STRUCTURE}/{DataSubCategories.EMISSION_FEES_EMISSION_TYPES}.csv"
    )
    emission_fee_df = pd.read_csv(
        csv_root_path
        / f"{DataCategories.SCENARIO}/scenario_1/{DataSubCategories.EMISSION_FEES}.csv"
    )
    return [emission_type_fee_df, emission_fee_df]


@pytest.fixture
def emission_fee_parser(emission_fee_dfs: list[pd.DataFrame]) -> EmissionFeeParser:
    return EmissionFeeParser(*emission_fee_dfs)


def test_create_emission_fee(
    emission_fee_parser: EmissionFeeParser, emission_fee_dfs: list[pd.DataFrame]
) -> None:
    emission_type_fee_df, emission_fee_df = emission_fee_dfs
    emission_fees = emission_fee_parser.create()

    assert all(isinstance(emf, EmissionFee) for emf in emission_fees)

    emission_fee_df.set_index("year_idx", inplace=True)
    assert len(emission_fees) == len(emission_fee_df.columns)
    assert list(emf.name for emf in emission_fees) == list(emission_fee_df.columns)
    assert all(emission_fee_df[emf.name].equals(emf.price) for emf in emission_fees)

    emission_type_fee_df = emission_type_fee_df.set_index("emission_fee").squeeze()
    assert all(
        emission_type_fee_df[emf.name] == emf.emission_type for emf in emission_fees
    )
