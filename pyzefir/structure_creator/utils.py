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

import json
from collections import defaultdict
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd

from pyzefir.structure_creator.constants_enums import XlsxFileName

T = TypeVar("T")
R = TypeVar("R")


def load_json(json_path: Path) -> dict:
    with open(json_path) as f:
        data = json.load(f)

    return data


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


def write_to_excel(
    data: dict,
    output_path: Path,
    filename: XlsxFileName | str,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / filename

    with pd.ExcelWriter(
        path=output_path,
        engine="openpyxl",
        mode="a" if output_path.is_file() else "w",
        if_sheet_exists="replace" if output_path.is_file() else None,
    ) as writer:
        for sheet_name, sheet_data in data.items():
            sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)


def invert_dict(dict_to_invert: dict[T, set[R] | list[R]]) -> dict[R, set[T]]:
    result_dict = defaultdict(set)
    for k, v_set in dict_to_invert.items():
        for v in v_set:
            result_dict[v].add(k)
    return dict(result_dict)


def merge_dicts(dict1: dict[T, set[R]], dict2: dict[T, set[R]]) -> dict[T, set[R]]:
    result = dict1.copy()
    for k, v in dict2.items():
        result[k] = result.get(k, set()) | v
    return result
