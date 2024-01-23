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

import pytest
from numpy import array

from pyzefir.model.network import NetworkElementsDict
from pyzefir.optimization.gurobi.preprocessing.indices import IndexingSet
from pyzefir.optimization.gurobi.preprocessing.parameters import ModelParameters
from tests.unit.optimization.gurobi.preprocessing.test_optimization_parameters.test_model_parameters.utils import (
    NetworkElementTestImplementation,
)


@pytest.mark.parametrize(
    ("element_names", "connections", "connected_element_names", "expected_result"),
    [
        (
            ["el1", "el2"],
            {"el1": "conn1", "el2": "conn2"},
            ["conn1", "conn5", "conn3", "conn2"],
            {0: 0, 1: 3},
        ),
        (["AA"], {"AA": "BB"}, ["X", "Y", "BB"], {0: 2}),
        (
            ["L1", "L2", "L3"],
            {"L1": "B1", "L2": "B2", "L3": "B3"},
            ["B1", "B2", "B3", "X", "Y"],
            {0: 0, 1: 1, 2: 2},
        ),
    ],
)
def test_get_index_from_prop(
    element_names: list[str],
    connections: dict[str, int],
    connected_element_names: list[str],
    expected_result: dict[int, int],
) -> None:
    element_idx, connected_element_idx = (
        IndexingSet(array(element_names)),
        IndexingSet(array(connected_element_names)),
    )
    network_elements = NetworkElementsDict(
        {
            name: NetworkElementTestImplementation(name=name, scalar_prop=idx)
            for name, idx in connections.items()
        }
    )

    assert (
        ModelParameters.get_index_from_prop(
            network_elements, element_idx, connected_element_idx, "scalar_prop"
        )
        == expected_result
    )
