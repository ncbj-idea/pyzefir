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
    """
    Generate a formatted name for a given lbs type and aggregate name.

    Args:
        - lbs_type (str): The type of low-emission source.
        - aggr_name (str): The name of the aggregate category associated with
          the low-emission source.

    Returns:
        - str: A formatted string that combines the aggregate name and lbs type,
          structured as "aggregate_name__lbs_type".
    """
    return f"{aggr_name}__{lbs_type}"


def interpolate_missing_df_values(
    values: pd.DataFrame, expected_idx: np.array
) -> pd.DataFrame:
    """
    Interpolate missing values in a DataFrame using linear interpolation.

    This function takes a DataFrame and an array of expected indices. It reindexes
    the DataFrame to align with the expected indices, filling in missing values
    with NaN. Linear interpolation is then applied to fill these NaN values based
    on the surrounding data points.

    Args:
        - values (pd.DataFrame): The DataFrame containing values to interpolate.
        - expected_idx (np.array): An array of expected indices that represent the
          desired time or sequential points for the DataFrame after interpolation.

    Returns:
        - pd.DataFrame: A new DataFrame with missing values linearly interpolated.
          If the original DataFrame matches the expected indices and contains no
          NaN values, it will be returned unchanged.
    """
    if (
        values.shape[0] == len(expected_idx)
        and np.array_equal(values.index, expected_idx)
        and not values.isnull().values.any()
    ):
        return values
    return values.reindex(index=expected_idx, fill_value=np.nan).interpolate(
        method="linear", axis=0
    )
