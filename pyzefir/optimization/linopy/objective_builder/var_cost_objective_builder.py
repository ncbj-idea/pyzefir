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
    """
    Class for building the variable cost objective.

    This class constructs the objective function representing the
    variable costs incurred by generators based on their fuel consumption
    and associated fuel costs. It aggregates the variable costs across
    all generators defined in the system.
    """

    def build_expression(self) -> LinearExpression | float:
        """
        Builds the variable cost objective.

        This method calculates the total variable cost by summing the
        variable costs for each generator that has an associated fuel type.
        If a generator does not have a fuel type, it is excluded from the
        total cost calculation.

        Returns:
            - LinearExpression | float: The total variable cost
              calculated from all generators' variable costs.
        """
        _logger.info("Building variable cost objective...")
        expr = 0.0
        for gen_idx in self.indices.GEN.ord:
            if self.parameters.gen.fuel[gen_idx] is not None:
                expr += self.generator_var_cost(gen_idx)
        _logger.info("Variable cost objective: Done")
        return expr

    def generator_var_cost(self, gen_idx: int) -> LinearExpression | float:
        """
        Calculates the variable cost for a given generator.

        This method computes the variable cost incurred by a specific
        generator based on its fuel consumption and the cost of the fuel.
        It forms a data array representing the cost per year and aggregates
        the total cost over the defined time period.

        Args:
            - gen_idx (int): The index of the generator for which the
              variable cost is being calculated.

        Returns:
            - LinearExpression | float: The calculated variable cost for
              the specified generator.
        """
        fuel_idx = self.parameters.gen.fuel[gen_idx]
        hourly_scale = self.parameters.scenario_parameters.hourly_scale
        cost = self.parameters.fuel.unit_cost[fuel_idx]
        fuel_consumption = self.expr.fuel_consumption(fuel_idx, gen_idx, hourly_scale)

        return (
            fuel_consumption
            * xr.DataArray(cost, dims=["year"], coords=[self.indices.Y.ii], name="cost")
            * self.indices.years_aggregation_array
        ).sum()
