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
import pandas as pd

from pyzefir.optimization.linopy.preprocessing.indices import IndexingSet, Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.linopy.utils import get_generators_capacity_multipliers
from pyzefir.optimization.results import Results


def objective_opex(
    indices: Indices, parameters: OptimizationParameters, results: Results
) -> float:
    return generator_opex(indices, parameters, results) + storage_opex(
        indices, parameters, results
    )


def generator_opex(
    indices: Indices, parameters: OptimizationParameters, results: Results
) -> float:
    return _opex_expression(
        indices.GEN,
        results.generators_results.cap,
        parameters.gen.tgen,
        parameters.tgen.opex,
        multipliers=get_generators_capacity_multipliers(
            parameters.scenario_parameters.generator_capacity_cost,
            parameters.tgen,
            parameters.gen,
        ),
        year_aggregation=indices._YEAR_AGGREGATION_DATA_ARRAY.to_numpy(),
    )


def storage_opex(
    indices: Indices, parameters: OptimizationParameters, results: Results
) -> float:
    return _opex_expression(
        indices.STOR,
        results.storages_results.cap,
        parameters.stor.tstor,
        parameters.tstor.opex,
        indices._YEAR_AGGREGATION_DATA_ARRAY.to_numpy(),
    )


def _opex_expression(
    unit_index: IndexingSet,
    cap: dict[str, pd.DataFrame],
    type_gen: dict,
    opex: np.ndarray,
    year_aggregation: np.ndarray,
    multipliers: dict[int, float] | None = None,
) -> float:
    result = 0.0
    for u_idx in unit_index.ord:
        result += (
            (
                opex[type_gen[u_idx]]
                * cap[unit_index.mapping[u_idx]]["cap"].values
                * year_aggregation
            ).sum()
            * multipliers[u_idx]
            if multipliers is not None
            else 1.0
        )
    return result
