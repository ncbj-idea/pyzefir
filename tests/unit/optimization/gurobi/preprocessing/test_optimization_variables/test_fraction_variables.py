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
from gurobipy import Model
from numpy import arange, array, ndarray

from pyzefir.model.network import Network
from pyzefir.optimization.gurobi.preprocessing.indices import IndexingSet, Indices
from pyzefir.optimization.gurobi.preprocessing.opt_variables import (
    OptimizationVariables,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.gurobi.conftest import N_YEARS


@pytest.mark.parametrize(
    ("y_sample", "n_aggr", "n_lbs"),
    [
        (array([0, 2]), 2, 3),
        (arange(N_YEARS), 1, 10),
        (array([0, 2, 4]), 1, 5),
        (array([0]), 5, 1),
    ],
)
def test_fraction_variables(
    y_sample: ndarray,
    n_aggr: int,
    n_lbs: int,
    opt_config: OptConfig,
    empty_network: Network,
) -> None:
    opt_config.year_sample = y_sample
    model, indices = Model(), Indices(empty_network, opt_config)
    indices.AGGR, indices.LBS = IndexingSet(arange(n_aggr)), IndexingSet(arange(n_lbs))
    variables = OptimizationVariables(model, indices, opt_config).frac
    model.update()

    assert variables.fraction.shape == (n_aggr, n_lbs, y_sample.shape[0])
