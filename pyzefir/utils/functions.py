from collections import defaultdict
from typing import TypeVar


def get_dict_vals(used_dict: dict[int, set] | None) -> set:
    """get all values of dictionary"""
    if used_dict is None:
        return set()
    return {element for value_set in used_dict.values() for element in value_set}


def _demch_unit(
    dmch_idx: int, dem_ch_tag: dict[int, int], unit_tags: dict[int, set[int]]
) -> set[int]:
    """unit idx of the corresponsing demand chunk id"""
    """dem_ch_tag: demand chunks idx to tag idx """
    """unit _tags: unit idx to tag idx"""
    return {
        gen_idx
        for gen_idx, tag_set in unit_tags.items()
        if dem_ch_tag[dmch_idx] in tag_set
    }


T = TypeVar("T")
Y = TypeVar("Y")


def invert_dict_of_sets(dict_: dict[T, set[Y]]) -> dict[Y, set[T]]:
    result_dict = defaultdict(set)
    for key, value_set in dict_.items():
        for value in value_set:
            result_dict[value].add(key)
    return dict(result_dict)
