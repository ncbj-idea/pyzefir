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

import numpy as np
import pandas as pd
import pytest
from numpy import all, arange, array, ones
from pandas import Series

from pyzefir.model.network import NetworkElementsDict
from pyzefir.optimization.gurobi.preprocessing.indices import IndexingSet
from pyzefir.optimization.gurobi.preprocessing.parameters import ModelParameters
from tests.unit.optimization.gurobi.constants import N_YEARS
from tests.unit.optimization.gurobi.preprocessing.test_optimization_parameters.test_model_parameters.utils import (
    EnergySourceTestImplementation,
    EnergySourceTypeTestImplementation,
)


def set_up(
    element_names: list[str],
    element_type: dict[str, str],
    data: dict[str, Any],
    prop_name: str,
    sample: Iterable | None,
) -> dict[int, Any]:
    element_idx = IndexingSet(array(element_names))
    elements = NetworkElementsDict(
        {
            name: EnergySourceTestImplementation(
                name=name,
                energy_source_type=type_name,
                unit_min_capacity=pd.Series([np.nan] * N_YEARS),
                unit_max_capacity=pd.Series([np.nan] * N_YEARS),
                unit_min_capacity_increase=pd.Series([np.nan] * N_YEARS),
                unit_max_capacity_increase=pd.Series([np.nan] * N_YEARS),
            )
            for name, type_name in element_type.items()
        }
    )
    element_types = NetworkElementsDict(
        {
            type_name: EnergySourceTypeTestImplementation(
                **{"name": type_name, prop_name: prop},
                min_capacity=pd.Series([np.nan] * N_YEARS),
                max_capacity=pd.Series([np.nan] * N_YEARS),
                min_capacity_increase=pd.Series([np.nan] * N_YEARS),
                max_capacity_increase=pd.Series([np.nan] * N_YEARS),
            )
            for type_name, prop in data.items()
        }
    )

    return ModelParameters.fetch_energy_source_type_prop(
        elements, element_types, element_idx, prop_name, sample
    )


@pytest.mark.parametrize(
    ("element_names", "element_type", "scalar_props", "sample", "expected_result"),
    [
        (
            ["e1", "e2", "e3", "e4", "e5"],
            {"e1": "t1", "e2": "t1", "e3": "t1", "e4": "t2", "e5": "t2"},
            {"t1": 10.5, "t2": -0.5},
            arange(200),
            {0: 10.5, 1: 10.5, 2: 10.5, 3: -0.5, 4: -0.5},
        ),
        (["e1"], {"e1": "t1"}, {"t1": 0.0, "t2": 2.3}, None, {0: 0.0}),
    ],
)
def test_fetch_element_type_prop_scalar_prop(
    element_names: list[str],
    element_type: dict[str, str],
    scalar_props: dict[str, float],
    sample: Iterable | None,
    expected_result: dict[int, float],
) -> None:
    result = set_up(element_names, element_type, scalar_props, "scalar_prop", sample)
    assert result == expected_result


@pytest.mark.parametrize(
    ("element_names", "element_type", "vector_props", "sample", "expected_result"),
    [
        (
            ["e1", "e2", "e3"],
            {"e1": "t1", "e2": "t1", "e3": "t2"},
            {"t1": Series(arange(100)), "t2": Series(ones(200))},
            arange(start=0, stop=100, step=2),
            {
                0: arange(start=0, stop=100, step=2),
                1: arange(start=0, stop=100, step=2),
                2: ones(50),
            },
        )
    ],
)
def test_fetch_element_type_prop_vector_prop(
    element_names: list[str],
    element_type: dict[str, str],
    vector_props: dict[str, Series],
    sample: Iterable | None,
    expected_result: dict[int, float],
) -> None:
    result = set_up(element_names, element_type, vector_props, "vector_prop", sample)
    assert set(result) == set(expected_result)
    for idx in result:
        assert all(result[idx] == expected_result[idx])


@pytest.mark.parametrize(
    ("element_names", "element_type", "dict_props", "sample", "expected_result"),
    [
        (
            ["e1", "e2", "e3"],
            {"e1": "t1", "e2": "t1", "e3": "t2"},
            {"t1": {"a": 1, "b": 2}, "t2": {"x": 666}},
            None,
            {0: {"a": 1, "b": 2}, 1: {"a": 1, "b": 2}, 2: {"x": 666}},
        )
    ],
)
def test_fetch_element_type_prop_dict_prop(
    element_names: list[str],
    element_type: dict[str, str],
    dict_props: dict[str, dict[str, int]],
    sample: Iterable | None,
    expected_result: dict[int, float],
) -> None:
    result = set_up(element_names, element_type, dict_props, "dict_prop", sample)
    assert result == expected_result
