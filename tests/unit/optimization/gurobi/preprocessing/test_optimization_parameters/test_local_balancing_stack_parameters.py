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

from pyzefir.model.network import Network, NetworkElementsDict
from pyzefir.model.network_elements import LocalBalancingStack
from pyzefir.optimization.gurobi.preprocessing.indices import IndexingSet, Indices
from pyzefir.optimization.gurobi.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.gurobi.preprocessing.parameters.local_balancing_stack_parameters import (
    LBSParameters,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.gurobi.preprocessing.utils import (
    aggregated_consumer_factory,
)


@pytest.mark.parametrize(
    ("lbs_names", "bus_names", "bus_out_data", "expected_result"),
    [
        (
            ["LBS1", "LBS2"],
            ["B3", "B1", "B2", "B4"],
            {"LBS1": {"heat": "B1", "ee": "B2"}, "LBS2": {"heat": "B3", "ee": "B4"}},
            {0: {"heat": 1, "ee": 2}, 1: {"heat": 0, "ee": 3}},
        ),
        (
            ["LBS1", "LBS2", "LBS3"],
            ["B1", "B2", "B3"],
            {"LBS1": {"ee": "B3"}, "LBS2": {"ee": "B1"}, "LBS3": {"ee": "B2"}},
            {0: {"ee": 2}, 1: {"ee": 0}, 2: {"ee": 1}},
        ),
        ([], ["B1", "B2", "B3"], {}, {}),
    ],
)
def test_get_bus_out(
    lbs_names: list[int],
    bus_names: list[int],
    bus_out_data: dict[str, dict[str, str]],
    expected_result: dict[int, dict[str, int]],
) -> None:
    stacks = NetworkElementsDict(
        {name: LocalBalancingStack(name, b_out) for name, b_out in bus_out_data.items()}
    )
    lbs_idx, bus_idx = IndexingSet(array(lbs_names)), IndexingSet(array(bus_names))

    result = LBSParameters.get_bus_out(stacks, bus_idx, lbs_idx)
    assert result == expected_result


@pytest.mark.parametrize(
    ("aggr_names", "lbs_names", "base_fractions", "expected_result"),
    [
        (
            ["AGGR1", "AGGR2"],
            ["LBS1", "LBS2", "LBS3", "LBS4", "LBS5"],
            {
                "AGGR1": {"LBS1": 0.0, "LBS2": 1.0},
                "AGGR2": {"LBS3": 0.3, "LBS4": 0.7, "LBS5": 0.0},
            },
            {0: 0, 1: 0, 2: 1, 3: 1, 4: 1},
        ),
        ([], ["LBS1", "LBS2"], {}, {}),
    ],
)
def test_get_aggr_idx(
    aggr_names: list[int],
    lbs_names: list[int],
    base_fractions: dict[str, dict[str, float]],
    expected_result: dict[int, int],
) -> None:
    aggregates = NetworkElementsDict(
        {
            name: aggregated_consumer_factory(name, stack_base_fraction=base_fr)
            for name, base_fr in base_fractions.items()
        }
    )
    aggr_idx, lbs_idx = IndexingSet(array(aggr_names)), IndexingSet(array(lbs_names))

    result = LBSParameters.get_aggr_idx(aggregates, aggr_idx, lbs_idx)
    assert result == expected_result


@pytest.mark.parametrize(
    ("lbs_names", "bus_names", "buses_data", "expected_result"),
    [
        (
            ["LBS1", "LBS2"],
            ["B1", "B2", "B3", "B4"],
            {
                "LBS1": {"ee": {"B1"}, "heat": {"B2"}},
                "LBS2": {"ee": {"B3"}, "heat": {"B4"}},
            },
            {
                0: {"ee": {0}, "heat": {1}},
                1: {"ee": {2}, "heat": {3}},
            },
        ),
        (
            ["LBS1", "LBS2", "LBS3"],
            ["B1", "B2", "B3", "B4", "B5", "B6"],
            {
                "LBS1": {"ee": {"B3", "B6"}},
                "LBS2": {"ee": {"B1", "B4"}},
                "LBS3": {"ee": {"B2", "B5"}},
            },
            {
                0: {"ee": {2, 5}},
                1: {"ee": {0, 3}},
                2: {"ee": {1, 4}},
            },
        ),
        ([], ["B1", "B2", "B3"], {}, {}),
    ],
)
def test_get_buses(
    lbs_names: list[int],
    bus_names: list[int],
    buses_data: dict[str, dict[str, set[str]]],
    expected_result: dict[int, dict[str, int]],
) -> None:
    stacks = NetworkElementsDict(
        {
            name: LocalBalancingStack(name=name, buses=b_out)
            for name, b_out in buses_data.items()
        }
    )
    lbs_idx, bus_idx = IndexingSet(array(lbs_names)), IndexingSet(array(bus_names))
    buses_result = LBSParameters.get_buses(stacks, bus_idx, lbs_idx)
    assert buses_result == expected_result


def test_create(complete_network: Network, opt_config: OptConfig) -> None:
    indices = Indices(complete_network, opt_config)
    lbs_params = OptimizationParameters(complete_network, indices, opt_config).lbs
    aggregates, stacks = (
        complete_network.aggregated_consumers,
        complete_network.local_balancing_stacks,
    )

    assert lbs_params.aggr_idx == LBSParameters.get_aggr_idx(
        aggregates, indices.AGGR, indices.LBS
    )
    assert lbs_params.bus_out == LBSParameters.get_bus_out(
        stacks, indices.BUS, indices.LBS
    )
    assert lbs_params.buses == LBSParameters.get_buses(stacks, indices.BUS, indices.LBS)
