# PyZefir
# Copyright (C) 2024 Narodowe Centrum Badań Jądrowych
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
from linopy import Model

from pyzefir.model.network import Network
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.linopy.preprocessing.opt_variables import (
    OptimizationVariables,
)
from pyzefir.optimization.opt_config import OptConfig


@pytest.mark.parametrize(
    ("h_sample", "y_sample", "bus_dsr_type"),
    [
        ([0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4], {0: 0}),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 1, 2, 3], {0: 0, 1: 1}),
    ],
)
def test_dsr_variables(
    h_sample: np.ndarray,
    y_sample: np.ndarray,
    bus_dsr_type: dict[int, int],
    complete_network: Network,
    opt_config: OptConfig,
) -> None:
    opt_config.hour_sample = h_sample
    opt_config.year_sample = y_sample
    indices = Indices(complete_network, opt_config)
    model = Model()
    parameters = OptimizationParameters(complete_network, indices, opt_config)
    parameters.bus.dsr_type = bus_dsr_type
    opt_var = OptimizationVariables(model, indices, opt_config)
    shift_minus_variables = opt_var.bus.shift_minus
    shift_plus_variables = opt_var.bus.shift_plus
    bus_dsr, yy, hh = (
        len(indices.BUS),
        len(opt_config.year_sample),
        len(opt_config.hour_sample),
    )
    assert shift_minus_variables.shape == (bus_dsr, hh, yy)
    assert shift_plus_variables.shape == (bus_dsr, hh, yy)
