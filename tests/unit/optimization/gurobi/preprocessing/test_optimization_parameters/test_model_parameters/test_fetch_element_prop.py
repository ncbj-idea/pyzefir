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

from typing import Any, Iterable

import pytest
from numpy import all, arange, array, ndarray, ones
from pandas import Series

from pyzefir.model.network import NetworkElementsDict
from pyzefir.optimization.gurobi.preprocessing.indices import IndexingSet
from pyzefir.optimization.gurobi.preprocessing.parameters import ModelParameters
from tests.unit.optimization.gurobi.preprocessing.test_optimization_parameters.test_model_parameters.utils import (
    NetworkElementTestImplementation,
)


def set_up(
    element_names: list[str], data: dict[str, Any], sample: Iterable, prop_name: str
) -> dict[int, Any]:
    test_elements_idx = IndexingSet(array(element_names))
    test_elements = NetworkElementsDict(
        {
            name: NetworkElementTestImplementation(**{"name": name, prop_name: value})
            for name, value in data.items()
        }
    )
    return ModelParameters.fetch_element_prop(
        test_elements, test_elements_idx, prop_name, sample
    )


@pytest.mark.parametrize(
    ("element_names", "scalar_props", "sample", "expected_result"),
    [
        (["element_1"], {"element_1": 4}, arange(1), {0: 4}),
        (
            ["el1", "el2", "el3"],
            {"el1": 0, "el2": 0, "el3": 0},
            range(25),
            {0: 0, 1: 0, 2: 0},
        ),
    ],
)
def test_fetch_element_prop_simple_prop(
    element_names: list[str],
    scalar_props: dict[str, int],
    sample: Iterable,
    expected_result: dict[str, int],
) -> None:
    result = set_up(element_names, scalar_props, sample, "scalar_prop")
    assert result == expected_result


@pytest.mark.parametrize(
    ("element_names", "vector_props", "sample", "expected_result"),
    [
        (
            ["el1", "el2"],
            {"el1": Series(arange(100)), "el2": Series(ones(100))},
            array([2, 4, 7, 10]),
            {0: array([2, 4, 7, 10]), 1: array([1, 1, 1, 1])},
        ),
        (
            ["e1", "e2", "e3"],
            {
                "e1": Series([1, 2, 3]),
                "e2": Series(arange(20)),
                "e3": Series(arange(10)),
            },
            None,
            {0: array([1, 2, 3]), 1: array(arange(20)), 2: array(arange(10))},
        ),
    ],
)
def test_fetch_element_prop_vector_prop(
    element_names: list[str],
    vector_props: dict[str, Series],
    sample: ndarray,
    expected_result: dict[int, Series],
) -> None:
    result = set_up(element_names, vector_props, sample, "vector_prop")
    assert set(result) == set(expected_result)
    for idx in result:
        assert all(result[idx] == expected_result[idx])


@pytest.mark.parametrize(
    ("element_names", "dict_props", "sample", "expected_result"),
    [
        (
            ["e1", "e2", "e3"],
            {"e1": {"heat": 1, "ee": 2}, "e2": {"a": 1, "b": 2, "c": 3}, "e3": {}},
            None,
            {0: {"heat": 1, "ee": 2}, 1: {"a": 1, "b": 2, "c": 3}, 2: {}},
        ),
        (
            ["e1", "e2", "e3"],
            {"e1": {"heat": 1, "ee": 2}, "e2": {"a": 1, "b": 2, "c": 3}, "e3": {}},
            arange(20),
            {0: {"heat": 1, "ee": 2}, 1: {"a": 1, "b": 2, "c": 3}, 2: {}},
        ),
    ],
)
def test_fetch_element_prop_dict_prop(
    element_names: list[str],
    dict_props: dict[str, dict[str, int]],
    sample: ndarray,
    expected_result: dict[int, dict[str, int]],
) -> None:
    result = set_up(element_names, dict_props, sample, "dict_prop")
    assert result == expected_result
