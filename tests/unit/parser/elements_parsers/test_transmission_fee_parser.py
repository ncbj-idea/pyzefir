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

from pyzefir.model.network_elements import TransmissionFee
from pyzefir.parser.elements_parsers.transmission_fee_parser import (
    TransmissionFeeParser,
)
from pyzefir.utils.path_manager import DataCategories, DataSubCategories


@pytest.fixture
def transmission_fees_dataframe(csv_root_path: Path) -> pd.DataFrame:
    transmission_fees = pd.read_csv(
        csv_root_path
        / f"{DataCategories.STRUCTURE}/{DataSubCategories.TRANSMISSION_FEES}.csv"
    )
    return transmission_fees


@pytest.fixture
def transmission_fee_parser(
    transmission_fees_dataframe: pd.DataFrame,
) -> TransmissionFeeParser:
    return TransmissionFeeParser(transmission_fees_dataframe)


def test_create_demand_profiles(
    transmission_fee_parser: TransmissionFeeParser,
    transmission_fees_dataframe: pd.DataFrame,
) -> None:
    transmission_fees_dataframe = transmission_fees_dataframe.set_index("hour_idx")
    transmission_fees = transmission_fee_parser.create()

    assert len(transmission_fees) == len(transmission_fees_dataframe.columns)
    assert all(isinstance(t, TransmissionFee) for t in transmission_fees)

    assert set(t.name for t in transmission_fees) == set(
        transmission_fees_dataframe.columns
    )

    assert all(
        transmission_fees_dataframe[t.name].equals(t.fee) for t in transmission_fees
    )
