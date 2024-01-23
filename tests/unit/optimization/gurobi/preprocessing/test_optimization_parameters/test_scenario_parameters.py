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
from numpy import all, arange, ndarray, ones

from pyzefir.model.network import Network
from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.gurobi.conftest import N_YEARS


@pytest.mark.parametrize(
    "discount", [ones(N_YEARS) * 0.05, ones(N_YEARS) * 0, arange(N_YEARS)]
)
def test_create(
    discount: ndarray, complete_network: Network, opt_config: OptConfig
) -> None:
    opt_config.discount_rate = discount
    indices = Indices(complete_network, opt_config)
    result = OptimizationParameters(
        complete_network, indices, opt_config
    ).scenario_parameters

    assert all(result.discount_rate == discount)
