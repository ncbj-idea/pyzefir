from collections import defaultdict
from typing import TypeVar

import numpy as np


def get_dict_vals(used_dict: dict[int, set] | None) -> set:
    """
    Retrieve all values from a dictionary of sets.

    This function collects all elements from the sets contained in the provided dictionary.
    If the dictionary is None, an empty set is returned.

    Args:
        - used_dict (dict[int, set] | None): A dictionary where each key maps to a set of values, or None.

    Returns:
        - set: A set containing all unique values from the sets in the dictionary,
            or an empty set if the input is None.
    """
    if used_dict is None:
        return set()
    return {element for value_set in used_dict.values() for element in value_set}


def demand_chunk_unit_indices(
    demand_chunk_idx: int,
    demand_chunk_tags: dict[int, int],
    unit_tags: dict[int, set[int]],
) -> set[int]:
    """
    Get the indices of units corresponding to a specific demand chunk id.

    This function identifies all unit indices that are associated with the tags of a given demand chunk.

    Args:
        - demand_chunk_idx (int): The index of the demand chunk.
        - demand_chunk_tags (dict[int, int]): A dictionary mapping demand chunk indices
            to their corresponding tag indices.
        - unit_tags (dict[int, set[int]]): A dictionary mapping unit indices to their
            corresponding sets of tag indices.

    Returns:
        - set[int]: A set of unit indices that are associated with the tags of the specified demand chunk.
    """
    return {
        unit_idx
        for unit_idx, tag_set in unit_tags.items()
        if demand_chunk_tags[demand_chunk_idx] in tag_set
    }


T = TypeVar("T")
Y = TypeVar("Y")


def invert_dict_of_sets(dict_: dict[T, set[Y]]) -> dict[Y, set[T]]:
    """
    Invert a dictionary mapping keys to sets of values.

    This function creates a new dictionary where each unique value from the sets in the input dictionary becomes a key,
    and the corresponding keys from the input dictionary are aggregated into sets.

    Args:
        - dict_ (dict[T, set[Y]]): A dictionary where each key maps to a set of values.

    Returns:
        - dict[Y, set[T]]: A new dictionary where each key is a value from the input sets,
            and each value is a set of keys from the input dictionary that mapped to it.
    """
    result_dict = defaultdict(set)
    for key, value_set in dict_.items():
        for value in value_set:
            result_dict[value].add(key)
    return dict(result_dict)


def is_flow_int(num: float | int | str | None) -> bool:
    """
    Check if a number is an integer or a float that is not NaN.

    This function determines if the provided argument is a number that can be considered a flow,
    which includes integers and floating-point numbers that are not NaN.

    Args:
        - num (float | int | str | None): The number to check.

    Returns:
        - bool: True if the number is an integer, a valid float,
            or an integer represented as a string; False otherwise.
    """
    if (
        num is not None
        and isinstance(num, float | int | np.integer)
        and not np.isnan(num)
    ):
        return True
    return False


def is_none_general(arg: dict | None) -> bool:
    """
    Determine if an argument is None or an empty dictionary.

    This function checks if the provided argument is either None or an empty dictionary.

    Args:
        - arg (dict | None): The argument to check.

    Returns:
        - bool: True if the argument is None or an empty dictionary; False otherwise.
    """
    res = False
    if isinstance(arg, dict):
        if len(arg) == 0:
            res = True
    elif arg is None:
        res = True
    return res


def flatten_list(list_of_lists: list[tuple[int]] | list[list[int]]) -> list[int]:
    """
    Flatten a list of lists or a list of tuples into a single list.

    Args:
        - list_of_lists (list[tuple[int]] | list[list[int]]): A list of lists or tuples containing integers.

    Returns:
        - list[int]: A flattened list containing all integer elements from the input.
    """
    return [elem for list_elem in list_of_lists for elem in list_elem]
