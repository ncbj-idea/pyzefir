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

import numpy as np
import pandas as pd


def get_lbs_name(lbs_type: str, aggr_name: str) -> str:
    return f"{aggr_name}__{lbs_type}"


def interpolate_missing_df_values(
    values: pd.DataFrame, expected_idx: np.array
) -> pd.DataFrame:
    if (
        values.shape[0] == len(expected_idx)
        and np.array_equal(values.index, expected_idx)
        and not values.isnull().values.any()
    ):
        return values
    return values.reindex(index=expected_idx, fill_value=np.nan).interpolate(
        method="linear", axis=0
    )
