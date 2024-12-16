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


class TransmissionFeeObjectiveBuilder(ObjectiveBuilder):
    """
    Class for building the transmission fee objective.

    This class constructs the objective function representing the
    transmission fees incurred by the energy lines. It aggregates the
    costs associated with the flow of energy across transmission lines
    based on defined transmission fees.
    """

    def build_expression(self) -> LinearExpression | float:
        """
        Builds the transmission fee objective.

        This method calculates the total transmission fee objective
        by summing the line flow costs for each transmission line defined
        in the parameters. If no transmission fees are specified, it
        returns zero.

        Returns:
            - LinearExpression | float: The total transmission fee
                objective, scaled by the hourly scenario parameter.
        """
        _logger.info("Building transmission fee objective...")
        if len(self.parameters.line.tf) == 0:
            return 0.0

        res = 0.0
        for line_idx, tf_idx in self.parameters.line.tf.items():
            res += self.line_flow_cost(line_idx, tf_idx).sum()
        _logger.info("Transmission fee objective: Done")
        return res * self.parameters.scenario_parameters.hourly_scale

    def line_flow_cost(self, line_idx: int, tf_idx: int) -> LinearExpression:
        """
        Calculates the line flow cost for a given transmission line.

        This method computes the cost associated with the flow of energy
        through a specific transmission line based on its transmission fee.
        It forms a data array representing the transmission fee applied
        to the line flow.

        Args:
            - line_idx (int): The index of the transmission line.
            - tf_idx (int): The index of the transmission fee.

        Returns:
            - LinearExpression: The calculated line flow cost for the
              specified transmission line.
        """
        transmission_fee = xr.DataArray(
            self.parameters.tf.fee[tf_idx],
            dims=["hour"],
            coords={
                "hour": self.indices.H.ii,
            },
        )
        line_flow = self.variables.line.flow.isel(line=line_idx)

        return line_flow * transmission_fee * self.indices.years_aggregation_array
