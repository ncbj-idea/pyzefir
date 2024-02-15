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

from typing import Iterable

import numpy as np
import pandas as pd

from pyzefir.optimization.linopy.preprocessing.indices import IndexingSet


def inv(name_list: Iterable[str], idx: IndexingSet) -> set[int]:
    return {idx.inverse[x] for x in name_list}


def _vectors_eq_base_compare(
    v1: np.ndarray | None | float,
    v2: pd.Series | None | float,
    sample: np.ndarray | None = None,
) -> bool:
    if isinstance(v2, pd.Series):
        return (
            v1 is not None
            and v2 is not None
            and bool(np.all(np.nan_to_num(v1) == np.nan_to_num(v2.values[sample])))
        )
    return False


def _vectors_eq_dict_compare(
    v1: dict,
    v2: dict,
) -> bool:
    if len(v1) == 0 and len(v2) == 0:
        return True
    return False


def _vectors_eq_float_compare(v1: float | None, v2: float | None) -> bool:
    return True if np.isnan(v1) and np.isnan(v2) else bool(v1 == v2)


def vectors_eq_check(
    v1: np.ndarray | None | float,
    v2: pd.Series | None | float,
    sample: np.ndarray | None = None,
) -> bool:
    if v1 is None and v2 is None:
        return True
    elif isinstance(v1, dict) and isinstance(v2, dict):
        return _vectors_eq_dict_compare(v1, v2)
    elif isinstance(v1, float) and isinstance(v2, float):
        return _vectors_eq_float_compare(v1, v2)
    return _vectors_eq_base_compare(v1, v2, sample)
