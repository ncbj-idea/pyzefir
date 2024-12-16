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
    """
    Class for building the emission fee objective in the energy system model.

    This class calculates the total cost associated with emissions produced by generators,
    applying emission fees based on fuel consumption and emission reduction parameters.
    The emission fee objective accounts for yearly generator emission costs, considering
    various emission types and their respective fees, adjusted by emission reductions.
    """

    def build_expression(self) -> LinearExpression | float:
        """
        Constructs the total emission fee objective across all generators and years.

        This method iterates through each generator and year to calculate the total cost
        incurred from emissions, using fuel consumption data and emission fees. The
        emission fee is computed by summing up the yearly emission costs for each generator,
        multiplied by scaling factors based on the year. If no valid fuel or emission fee
        data is available, the method returns 0.0.

        Returns:
            - LinearExpression | float: The total emission fee objective, or 0.0 if no fees apply.
        """
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
        """
        Constructs the emission cost for a specific generator in a specific year.

        This method calculates the yearly emission cost for a given generator by using
        fuel consumption data and applying emission fees based on the type of emissions
        generated. It considers any emission reductions that apply to the generator and
        calculates the total cost for each type of emission produced in the given year.

        Args:
            - year_idx (int): The index of the year for which to calculate emission costs.
            - gen_idx (int): The index of the generator for which to calculate emission costs.
            - eh (ExpressionHandler): The expression handler used to manage fuel consumption data.

        Returns:
            - LinearExpression | float: The emission cost for the generator in the specified year or 0.0.
        """
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
