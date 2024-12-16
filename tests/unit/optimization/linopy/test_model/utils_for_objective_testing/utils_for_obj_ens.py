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

from pyzefir.optimization.linopy.objective_builder.ens_penalty_builder import (
    EnsPenaltyCostObjectiveBuilder,
)
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.results import Results


def objective_ens(
    indices: Indices, parameters: OptimizationParameters, results: Results
) -> float:
    if not parameters.scenario_parameters.ens_penalty_cost:
        return 0.0

    et_penalty_cost = EnsPenaltyCostObjectiveBuilder(
        indices, parameters, None, None
    )._get_ens_penalty()

    expr = 0.0
    for et, penalty_cost in et_penalty_cost.items():
        bus_et = np.array(
            list(
                {
                    indices.BUS.mapping[bus]
                    for bus, e_type in parameters.bus.et.items()
                    if et == e_type
                }
            )
        )
        expr += sum(
            sum(results.bus_results.bus_ens[bus][year].sum() for bus in bus_et)
            * penalty_cost
            * indices._YEAR_AGGREGATION_DATA_ARRAY.to_numpy()[year]
            for year in indices.Y.ii
        )

    return expr
