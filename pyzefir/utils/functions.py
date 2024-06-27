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

from collections import defaultdict
from typing import TypeVar

import numpy as np


def get_dict_vals(used_dict: dict[int, set] | None) -> set:
    """get all values of dictionary"""
    if used_dict is None:
        return set()
    return {element for value_set in used_dict.values() for element in value_set}


def demand_chunk_unit_indices(
    demand_chunk_idx: int,
    demand_chunk_tags: dict[int, int],
    unit_tags: dict[int, set[int]],
) -> set[int]:
    """unit idx of the corresponding demand chunk id"""
    """dem_ch_tag: demand chunks idx to tag idx """
    """unit _tags: unit idx to tag idx"""
    return {
        unit_idx
        for unit_idx, tag_set in unit_tags.items()
        if demand_chunk_tags[demand_chunk_idx] in tag_set
    }


T = TypeVar("T")
Y = TypeVar("Y")


def invert_dict_of_sets(dict_: dict[T, set[Y]]) -> dict[Y, set[T]]:
    result_dict = defaultdict(set)
    for key, value_set in dict_.items():
        for value in value_set:
            result_dict[value].add(key)
    return dict(result_dict)


def is_flow_int(num: float | int | str | None) -> bool:
    """check if general float number allowing int type"""
    if (
        num is not None
        and isinstance(num, float | int | np.integer)
        and not np.isnan(num)
    ):
        return True
    return False


def is_none_general(arg: dict | None) -> bool:
    """returns True if None or empty dict"""
    res = False
    if isinstance(arg, dict):
        if len(arg) == 0:
            res = True
    elif arg is None:
        res = True
    return res


def flatten_list(list_of_lists: list[tuple[int]] | list[list[int]]) -> list[int]:
    return [elem for list_elem in list_of_lists for elem in list_elem]
