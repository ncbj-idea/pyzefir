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

from gurobipy import MLinExpr, quicksum

from pyzefir.optimization.gurobi.objective_builder import ObjectiveBuilder


class VarCostObjectiveBuilder(ObjectiveBuilder):
    def build_expression(self) -> MLinExpr:
        return quicksum(
            self.generator_var_cost(gen_idx)
            for gen_idx in self.indices.GEN.ord
            if self.parameters.gen.fuel[gen_idx] is not None
        ).sum()

    def generator_var_cost(self, gen_idx: int) -> MLinExpr | float:
        fuel_idx = self.parameters.gen.fuel[gen_idx]
        hourly_scale = self.parameters.scenario_parameters.hourly_scale
        cost = self.parameters.fuel.unit_cost[fuel_idx]
        fuel_consumption = self.expr.fuel_consumption(fuel_idx, gen_idx, hourly_scale)

        return fuel_consumption * cost
