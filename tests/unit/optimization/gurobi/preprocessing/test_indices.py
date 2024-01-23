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

import numpy as np

from pyzefir.model.network import Network
from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.opt_config import OptConfig


def network_elements_indices_test(network: Network, indices: Indices) -> None:
    test_cases = [
        (network.fuels, indices.FUEL),
        (network.capacity_factors, indices.CF),
        (network.generators, indices.GEN),
        (network.buses, indices.BUS),
        (network.storages, indices.STOR),
        (network.lines, indices.LINE),
        (network.local_balancing_stacks, indices.LBS),
        (network.aggregated_consumers, indices.AGGR),
    ]

    assert set(indices.ET.ii) == set(network.energy_types)
    assert np.all(indices.ET.ord == np.arange(len(network.energy_types)))

    for elements_dict, indexing_set in test_cases:
        expected_length = len(elements_dict)
        assert np.all(indexing_set.ord == np.arange(expected_length))
        assert set(indexing_set.ii) == set(
            [element.name for element in elements_dict.values()]
        )


def test_create_indices_on_empty_network(
    empty_network: Network, opt_config: OptConfig
) -> None:
    indices = Indices(empty_network, opt_config)
    network_elements_indices_test(empty_network, indices)


def test_create_indices_on_complete_network(
    complete_network: Network, opt_config: OptConfig
) -> None:
    indices = Indices(complete_network, opt_config)
    network_elements_indices_test(complete_network, indices)
