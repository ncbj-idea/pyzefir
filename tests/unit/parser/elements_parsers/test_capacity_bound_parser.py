# PyZefir
# Copyright (C) 2024 Narodowe Centrum Badań Jądrowych
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

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.network_elements.capacity_bound import CapacityBound
from pyzefir.parser.elements_parsers.capacity_bound_parser import CapacityBoundParser
from pyzefir.utils.path_manager import DataCategories, DataSubCategories


@pytest.fixture
def capacity_bounds_df(csv_root_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_root_path
        / f"{DataCategories.SCENARIO}"
        / "scenario_1"
        / f"{DataSubCategories.CAPACITY_BOUNDS}.csv"
    )


def test_create_capacity_bounds(capacity_bounds_df: pd.DataFrame) -> None:
    capacity_bounds = CapacityBoundParser(capacity_bounds_df).create()
    target_name = "capa_bounds_gen_stor"

    assert isinstance(capacity_bounds, tuple)
    assert all(isinstance(cb, CapacityBound) for cb in capacity_bounds)
    assert len(capacity_bounds) == capacity_bounds_df.shape[0]

    capacity_bounds_df = capacity_bounds_df.set_index("name")
    assert np.isnan(capacity_bounds_df.loc[target_name, "left_coeff"])
    assert {obj.name: obj for obj in capacity_bounds}[
        target_name
    ].left_coefficient == pytest.approx(1.0)
