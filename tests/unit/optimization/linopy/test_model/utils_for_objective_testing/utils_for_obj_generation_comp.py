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


def objective_generation_compensation(
    indices: Indices, parameters: OptimizationParameters, results: Results
) -> float:
    gen_to_type_dict = {
        k: v
        for k, v in parameters.gen.tgen.items()
        if v in parameters.tgen.generation_compensation.keys()
    }
    hourly_scale = parameters.scenario_parameters.hourly_scale
    unit_map = indices.GEN.mapping
    expr = 0.0
    for gen_idx, tgen_idx in gen_to_type_dict.items():
        compensation = parameters.tgen.generation_compensation[tgen_idx]
        generation = np.asarray(
            results.generators_results.gen[unit_map[gen_idx]].sum(axis=0)
        )
        expr -= np.dot(compensation, generation) * hourly_scale
    return expr
