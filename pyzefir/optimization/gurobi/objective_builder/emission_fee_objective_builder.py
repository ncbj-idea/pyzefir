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

from pyzefir.optimization.gurobi.expression_handler import ExpressionHandler
from pyzefir.optimization.gurobi.objective_builder import ObjectiveBuilder


class EmissionFeeObjectiveBuilder(ObjectiveBuilder):
    def build_expression(self) -> MLinExpr | float:
        if (
            not self.parameters.gen.fuel
            or all(not fuel for fuel in self.parameters.gen.fuel.values())
            or all(not em_fee for em_fee in self.parameters.gen.emission_fee.values())
        ):
            return 0.0
        else:
            eh = ExpressionHandler(self.indices, self.variables, self.parameters)
            return quicksum(
                self.yearly_generator_emission_cost(
                    year_idx=year_idx, gen_idx=gen_idx, eh=eh
                )
                for year_idx in self.indices.Y.ord
                for gen_idx in self.indices.GEN.ord
            ).sum()

    def yearly_generator_emission_cost(
        self, year_idx: int, gen_idx: int, eh: ExpressionHandler
    ) -> MLinExpr | float:
        fuel_idx = self.parameters.gen.fuel[gen_idx]
        if fuel_idx is None:
            return 0.0
        fc = eh.fuel_consumption(
            fuel_idx, gen_idx, self.parameters.scenario_parameters.hourly_scale
        )[year_idx]
        total_emission = 0.0
        for emission_fee_idx in self.parameters.gen.emission_fee[gen_idx]:
            emission_type = self.parameters.emf.emission_type[emission_fee_idx]
            generator_emission = (
                fc
                * self.parameters.fuel.u_emission[fuel_idx][emission_type]
                * (1 - self.parameters.gen.em_red[gen_idx][emission_type])
            )
            total_emission += (
                generator_emission
                * self.parameters.emf.price[emission_fee_idx][year_idx]
            )
        return total_emission
