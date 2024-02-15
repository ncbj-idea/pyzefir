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
from linopy import Model
from numpy import arange, array, ndarray

from pyzefir.model.network import Network
from pyzefir.optimization.linopy.preprocessing.indices import IndexingSet, Indices
from pyzefir.optimization.linopy.preprocessing.opt_variables import (
    OptimizationVariables,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.conftest import N_YEARS


@pytest.mark.parametrize(
    ("y_sample", "h_sample", "n_line"),
    [
        (arange(N_YEARS), arange(50), 1),
        (arange(N_YEARS), arange(500), 10),
        (array([1]), array([1]), 10),
        (arange(2), arange(100), 4),
    ],
)
def test_line_variables(
    y_sample: ndarray,
    h_sample: ndarray,
    n_line: int,
    opt_config: OptConfig,
    empty_network: Network,
) -> None:
    opt_config.year_sample = y_sample
    opt_config.hour_sample = h_sample
    model, indices = Model(), Indices(empty_network, opt_config)
    indices.LINE = IndexingSet(arange(n_line))
    variables = OptimizationVariables(model, indices, opt_config).line

    assert variables.flow.shape == (n_line, h_sample.shape[0], y_sample.shape[0])
