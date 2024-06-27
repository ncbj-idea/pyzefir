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

from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.results import Results


def objective_varcost(
    indices: Indices, parameters: OptimizationParameters, results: Results
) -> float:
    return sum(
        _generator_var_cost(indices, parameters, results, gen_idx).sum()
        for gen_idx in indices.GEN.ord
        if parameters.gen.fuel[gen_idx] is not None
    ) + sum(
        _transmission_var_cost(indices, parameters, results, line_idx).sum()
        for line_idx in indices.LINE.ord
    )


def _transmission_var_cost(
    indices: Indices,
    parameters: OptimizationParameters,
    results: Results,
    line_idx: int,
) -> np.ndarray:
    line_map = indices.LINE.mapping
    hourly_scale = parameters.scenario_parameters.hourly_scale

    return np.asarray(
        [
            (
                results.lines_results.flow[line_map[line_idx]][year]
                * parameters.tf.fee[parameters.line.tf[line_idx]]
            ).sum()
            * hourly_scale
            * indices._YEAR_AGGREGATION_DATA_ARRAY.to_numpy()[year]
            for year in results.lines_results.flow[line_map[line_idx]].columns
            if line_idx in parameters.line.tf
        ]
    )


def _generator_var_cost(
    indices: Indices, parameters: OptimizationParameters, results: Results, gen_idx: int
) -> np.ndarray:
    fuel_idx = parameters.gen.fuel[gen_idx]
    hourly_scale = parameters.scenario_parameters.hourly_scale
    cost = parameters.fuel.unit_cost[fuel_idx]
    fuel_consumption = _fuel_consumption(
        indices, parameters, results, fuel_idx, gen_idx, hourly_scale
    )

    return (
        np.asarray(fuel_consumption * cost)
        * indices._YEAR_AGGREGATION_DATA_ARRAY.to_numpy()
    )


def _fuel_consumption(
    indices: Indices,
    parameters: OptimizationParameters,
    results: Results,
    fuel_idx: int,
    gen_idx: int,
    hourly_scale: float,
) -> pd.Series | np.ndarray:
    """Fuel consumption

    Args:
        fuel_idx (int): fuel index
        gen_idx (int): generator index
        hourly_scale: (float): hourly scale
    Returns:
        fuel consumption multiply by hourly scale
    """

    if parameters.gen.fuel[gen_idx] != fuel_idx:
        return np.zeros(len(indices.Y))
    unit_map = indices.GEN.mapping
    return (
        results.generators_results.gen[unit_map[gen_idx]].sum(axis=0)
        / parameters.fuel.energy_per_unit[fuel_idx]
    ) * hourly_scale
