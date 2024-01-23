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
from gurobipy import Model

from pyzefir.model.network import Network
from pyzefir.model.network_elements import DemandChunk
from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.opt_variables import (
    OptimizationVariables,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.gurobi.test_model.utils import (
    set_network_elements_parameters,
)


@pytest.mark.parametrize(
    (
        "h_sample",
        "y_sample",
    ),
    [
        (
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4],
        ),
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [0, 1, 2, 3],
        ),
    ],
)
def test_demand_chunks_variables(
    h_sample: np.ndarray,
    y_sample: np.ndarray,
    complete_network: Network,
    opt_config: OptConfig,
) -> None:
    periods = np.array([[0, 100], [101, 8760]])
    example_demand = np.array([[20, 20], [20000, 20000]])
    complete_network.demand_chunks = {
        "test_1": DemandChunk(
            name="test_1",
            demand=example_demand,
            energy_type="electricity",
            periods=periods,
            tag="ee_tag",
        )
    }
    unit_tags = {"chp_coal_grid_hs": ["ee_tag", "heat_tag"]}
    for unit, unit_tag in unit_tags.items():
        set_network_elements_parameters(
            complete_network.generators, {unit: {"tags": unit_tag}}
        )
    opt_config.hour_sample = h_sample
    opt_config.year_sample = y_sample
    indices = Indices(complete_network, opt_config)
    model = Model()

    gen_variables = OptimizationVariables(model, indices, opt_config).gen
    gen_storages = OptimizationVariables(model, indices, opt_config).stor
    ets, yy, hh = (
        len(complete_network.energy_types),
        len(opt_config.year_sample),
        len(opt_config.hour_sample),
    )
    dmch = len(complete_network.demand_chunks)
    n_gen, n_stor = len(complete_network.generators), len(complete_network.storages)
    assert gen_variables.gen_dch.shape == (ets, dmch, n_gen, hh, yy)
    assert gen_storages.gen_dch.shape == (dmch, n_stor, hh, yy)
