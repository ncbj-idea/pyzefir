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

from linopy import LinearExpression

from pyzefir.optimization.linopy.expression_handler import ExpressionHandler
from pyzefir.optimization.linopy.objective_builder import ObjectiveBuilder

_logger = logging.getLogger(__name__)


class EmissionFeeObjectiveBuilder(ObjectiveBuilder):
    def build_expression(self) -> LinearExpression | float:
        _logger.info("Building emission fee objective...")
        result = 0.0
        if not (
            not self.parameters.gen.fuel
            or all(not fuel for fuel in self.parameters.gen.fuel.values())
            or all(not em_fee for em_fee in self.parameters.gen.emission_fee.values())
        ):
            eh = ExpressionHandler(self.indices, self.variables, self.parameters)
            for gen_idx in self.indices.GEN.ord:
                for year_idx in self.indices.Y.ord:
                    result += self.yearly_generator_emission_cost(
                        year_idx=year_idx, gen_idx=gen_idx, eh=eh
                    ) * self.indices.year_aggregates(year_idx)
        _logger.info("Emission fee objective: Done")
        return result

    def yearly_generator_emission_cost(
        self, year_idx: int, gen_idx: int, eh: ExpressionHandler
    ) -> LinearExpression | float:
        fuel_idx = self.parameters.gen.fuel[gen_idx]
        if fuel_idx is None:
            return 0.0
        fc = eh.fuel_consumption(
            fuel_idx, gen_idx, self.parameters.scenario_parameters.hourly_scale
        ).isel(year=year_idx)
        total_emission = 0.0
        for emission_fee_idx in self.parameters.gen.emission_fee[gen_idx]:
            emission_type = self.parameters.emf.emission_type[emission_fee_idx]
            generator_emission = (
                fc
                * self.parameters.fuel.u_emission[fuel_idx][emission_type]
                * (1 - self.parameters.gen.em_red[gen_idx][emission_type][year_idx])
            )
            total_emission += (
                generator_emission
                * self.parameters.emf.price[emission_fee_idx][year_idx]
            )
        return total_emission
