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
from pyzefir.optimization.linopy.utils import get_generators_capacity_multipliers

_logger = logging.getLogger(__name__)


class OpexObjectiveBuilder(ObjectiveBuilder):
    """
    Class for building the operational expenditure (opex) objective.

    This class constructs the objective function representing the total
    operational costs for generators and storage facilities. It computes
    the operational expenses associated with the generation and storage
    of energy, facilitating cost optimization in energy systems.
    """

    def build_expression(self) -> LinearExpression:
        """
        Builds the opex objective for generators and storage facilities.

        This method aggregates the operational expenditures from both
        generators and storage units into a single objective expression
        that can be used for optimization purposes.

        Returns:
            - LinearExpression: The total opex objective for the system.
        """
        _logger.info("Building opex objective...")
        generators_opex = self.generator_opex()
        storages_opex = self.storage_opex()
        return generators_opex + storages_opex

    def generator_opex(self) -> LinearExpression:
        """
        Builds the opex objective specifically for generators.

        This method calculates the operational expenditures for all
        generators based on their individual operating costs and capacity
        multipliers. It forms a data array representing the opex for
        each generator across different years.

        Returns:
            - LinearExpression: The calculated opex for generators.
        """
        multipliers = get_generators_capacity_multipliers(
            self.parameters.scenario_parameters.generator_capacity_cost,
            self.parameters.tgen,
            self.parameters.gen,
        )
        opex = xr.DataArray(
            [
                self.parameters.tgen.opex[self.parameters.gen.tgen[gen_idx]]
                * multipliers[gen_idx]
                for gen_idx in self.indices.GEN.ord
            ],
            dims=["gen", "year"],
            coords=[self.indices.GEN.ii, self.indices.Y.ii],
            name="opex",
        )
        _logger.info("Building generator opex expression: Done")
        return (
            opex * self.variables.gen.cap * self.indices.years_aggregation_array
        ).sum()

    def storage_opex(self) -> LinearExpression | float:
        """
        Builds the opex objective for storage facilities.

        This method calculates the operational expenditures for all
        storage units if they exist. It forms a data array representing
        the opex for each storage unit across different years.

        Returns:
            - LinearExpression | float: The calculated opex for storage
                units, or 0.0 if no storage units are defined.
        """
        if self.indices.STOR.ord.size:
            opex = xr.DataArray(
                [
                    self.parameters.tstor.opex[self.parameters.stor.tstor[stor_idx]]
                    for stor_idx in self.indices.STOR.ord
                ],
                dims=["stor", "year"],
                coords=[self.indices.STOR.ii, self.indices.Y.ii],
                name="opex",
            )
            _logger.info("Building generator opex expression: Done")
            return (
                opex * self.variables.stor.cap * self.indices.years_aggregation_array
            ).sum()
        _logger.warning("Size of storage not set, returning default expression.")
        return 0
