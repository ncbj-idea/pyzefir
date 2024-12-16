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

from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.results import Results


def objective_curtailment_cost(
    indices: Indices, parameters: OptimizationParameters, results: Results
) -> float:
    hourly_scale = parameters.scenario_parameters.hourly_scale
    curtailment_cost = parameters.tgen.energy_curtailment_cost
    gen_idx_to_tgen_idx = {
        gen_idx: tgen_idx
        for gen_idx, tgen_idx in parameters.gen.tgen.items()
        if tgen_idx in curtailment_cost
    }
    result = 0.0
    for gen_idx, tgen_idx in gen_idx_to_tgen_idx.items():
        for et in parameters.gen.ett[gen_idx]:
            dump_et = (
                results.generators_results.dump_et[indices.GEN.mapping[gen_idx]][et]
                * indices._YEAR_AGGREGATION_DATA_ARRAY.to_numpy()
            ).sum(axis=0)
            result = dump_et @ curtailment_cost[tgen_idx] * hourly_scale + result

    return result
