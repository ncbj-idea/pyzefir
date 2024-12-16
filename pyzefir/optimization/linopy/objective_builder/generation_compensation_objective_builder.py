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


class GenerationCompensationObjectiveBuilder(ObjectiveBuilder):
    """
    Class for building the generation compensation objective.

    This class is responsible for calculating the compensation for electricity
    generation from different types of generators. It constructs an objective
    expression that accounts for the compensation amounts based on generator
    outputs and their associated compensation rates defined in the system parameters.
    """

    def build_expression(self) -> LinearExpression | float:
        """
        Builds the generation compensation objective expression.

        This method compiles the total compensation for all generators that are
        eligible for generation compensation. It iterates through the generators,
        retrieving their respective compensation values and summing them to form
        the objective expression.

        Returns:
            - LinearExpression | float: The total generation compensation expression.
        """
        _logger.info("Building generation compensation objective...")
        gen_to_type_dict = {
            k: v
            for k, v in self.parameters.gen.tgen.items()
            if v in self.parameters.tgen.generation_compensation.keys()
        }
        expr = sum(
            [
                self.generator_compensation(gen_idx, tgen_idx)
                for gen_idx, tgen_idx in gen_to_type_dict.items()
            ]
        )
        _logger.info("Variable generation compensation objective: Done")
        return expr

    def generator_compensation(
        self, gen_idx: int, tgen_idx: int
    ) -> LinearExpression | float:
        """
        Calculates the compensation for a given generator.

        This method computes the compensation amount for a specific generator
        based on its output and the compensation rate associated with its
        generator type. The compensation is scaled by the hourly production
        and aggregated over the defined years.

        Args:
            - gen_idx (int): Index of the generator for which compensation is calculated.
            - tgen_idx (int): Index of the compensation generator type.

        Returns:
            - LinearExpression | float: The calculated compensation for the generator.
        """
        hourly_scale = self.parameters.scenario_parameters.hourly_scale
        compensation = self.parameters.tgen.generation_compensation[tgen_idx]
        generation = self.variables.gen.gen.isel(gen=gen_idx).sum(["hour"])
        return (
            -generation
            * xr.DataArray(
                compensation,
                dims=["year"],
                coords=[self.indices.Y.ii],
                name="compensation",
            )
            * self.indices.years_aggregation_array
        ).sum() * hourly_scale
