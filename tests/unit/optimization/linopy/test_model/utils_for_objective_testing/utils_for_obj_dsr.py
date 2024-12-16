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

from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.results import Results


def objective_dsr(
    indices: Indices, parameters: OptimizationParameters, results: Results
) -> float:
    if len(parameters.bus.dsr_type):
        shift_minus = results.bus_results.shift_minus
        shift_plus = results.bus_results.shift_plus
        penalization = parameters.dsr.penalization_minus
        penalization_plus = parameters.dsr.penalization_plus
        bus_parameters = parameters.bus
        return sum(
            np.asarray(
                shift_minus[indices.BUS.mapping[bus_idx]]
                * indices._YEAR_AGGREGATION_DATA_ARRAY.to_numpy()
            ).sum()
            * penalization[dsr_idx]
            + np.asarray(
                shift_plus[indices.BUS.mapping[bus_idx]]
                * indices._YEAR_AGGREGATION_DATA_ARRAY.to_numpy()
            ).sum()
            * penalization_plus[dsr_idx]
            for bus_idx, dsr_idx in bus_parameters.dsr_type.items()
        )
    else:
        return 0.0
