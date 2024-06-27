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
import logging

import xarray as xr
from linopy import LinearExpression

from pyzefir.optimization.linopy.objective_builder import ObjectiveBuilder

_logger = logging.getLogger(__name__)


class VarCostObjectiveBuilder(ObjectiveBuilder):
    def build_expression(self) -> LinearExpression | float:
        _logger.info("Building variable cost objective...")
        expr = 0.0
        for gen_idx in self.indices.GEN.ord:
            if self.parameters.gen.fuel[gen_idx] is not None:
                expr += self.generator_var_cost(gen_idx)
        _logger.info("Variable cost objective: Done")
        return expr

    def generator_var_cost(self, gen_idx: int) -> LinearExpression | float:
        fuel_idx = self.parameters.gen.fuel[gen_idx]
        hourly_scale = self.parameters.scenario_parameters.hourly_scale
        cost = self.parameters.fuel.unit_cost[fuel_idx]
        fuel_consumption = self.expr.fuel_consumption(fuel_idx, gen_idx, hourly_scale)

        return (
            fuel_consumption
            * xr.DataArray(cost, dims=["year"], coords=[self.indices.Y.ii], name="cost")
            * self.indices.years_aggregation_array
        ).sum()
