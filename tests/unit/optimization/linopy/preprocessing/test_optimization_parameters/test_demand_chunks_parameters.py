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
import pytest

from pyzefir.model.network import Network
from pyzefir.model.network_elements import DemandChunk
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.test_model.utils import (
    set_network_elements_parameters,
)


@pytest.mark.parametrize(
    (
        "energy_type",
        "expected_energy_type",
        "tag",
        "expected_tag",
        "periods",
        "expected_periods",
        "demand",
        "expected_demand",
    ),
    [
        (
            "electricity",
            {0: "electricity"},
            "ee_tag",
            {0: 0},
            np.array([[0, 100], [101, 8760]]),
            {0: np.array([[0, 100], [101, 8760]])},
            np.array([[20, 20, 20, 20, 20], [20000, 20000, 20000, 20000, 20000]]),
            {0: np.array([[20, 20, 20, 20, 20], [20000, 20000, 20000, 20000, 20000]])},
        ),
        (
            "heat",
            {0: "heat"},
            "heat_tag",
            {0: 1},
            np.array([[0, 100], [101, 8760]]),
            {0: np.array([[0, 100], [101, 8760]])},
            np.array([[20, 20, 20], [20000, 20000, 20000]]),
            {0: np.array([[20, 20, 20], [20000, 20000, 20000]])},
        ),
    ],
)
def test_demand_chunks_parameters(
    energy_type: str,
    expected_energy_type: dict[int, str],
    tag: str,
    expected_tag: dict[int, int],
    periods: np.ndarray,
    expected_periods: dict[int, np.ndarray],
    demand: np.ndarray,
    expected_demand: dict[int, np.ndarray],
    complete_network: Network,
    opt_config: OptConfig,
) -> None:
    complete_network.demand_chunks = {
        "test_1": DemandChunk(
            name="test_1",
            demand=demand,
            energy_type=energy_type,
            periods=periods,
            tag=tag,
        )
    }
    unit_tags = {"chp_coal_grid_hs": ["ee_tag", "heat_tag"]}
    for unit, unit_tag in unit_tags.items():
        set_network_elements_parameters(
            complete_network.generators, {unit: {"tags": unit_tag}}
        )
    indices = Indices(complete_network, opt_config)
    params = OptimizationParameters(
        complete_network, indices, opt_config
    ).demand_chunks_parameters
    assert params.energy_type == expected_energy_type
    assert params.tag == expected_tag
    assert np.all(params.periods[0] == expected_periods[0])
    assert np.all(params.demand[0] == expected_demand[0])
