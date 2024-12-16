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

from pyzefir.optimization.linopy.objective_builder import ObjectiveBuilder

_logger = logging.getLogger(__name__)


class DsrPenaltyObjectiveBuilder(ObjectiveBuilder):
    """
    Class for building the objective function related to demand-side response (DSR) penalties.

    This class is responsible for constructing the penalty objective tied to demand-side
    response (DSR) mechanisms in an energy system. DSR penalties are applied when energy
    demand is shifted either upwards or downwards from the baseline, penalizing deviations
    based on predefined parameters.
    """

    def build_expression(self) -> LinearExpression | float:
        """
        Constructs the DSR penalty objective expression.

        This method builds the objective function for DSR penalties by calculating the
        penalties for both upward (shift_plus) and downward (shift_minus) shifts in energy
        demand for buses that participate in DSR. The penalty is calculated based on the
        shifting variables and associated penalization parameters for each DSR type. If no
        DSR types are specified, a default value of 0.0 is returned.

        Returns:
            - LinearExpression | float: The total penalty expression for DSR, or 0.0 if DSR is not specified.
        """
        _logger.info("Building DSR penalty objective...")
        if self.parameters.bus.dsr_type:
            shift_minus = self.variables.bus.shift_minus
            shift_plus = self.variables.bus.shift_plus
            penalization = self.parameters.dsr.penalization_minus
            penalization_plus = self.parameters.dsr.penalization_plus
            bus_parameters = self.parameters.bus
            result: LinearExpression = 0.0
            for bus_idx, dsr_idx in bus_parameters.dsr_type.items():
                result += (
                    shift_minus[bus_idx] * self.indices.years_aggregation_array
                ).sum() * penalization[dsr_idx] + (
                    shift_plus[bus_idx] * self.indices.years_aggregation_array
                ).sum() * penalization_plus[
                    dsr_idx
                ]
            _logger.info("DSR penalty objective: Done")
            return result.sum()
        _logger.warning("No specified DSR type, returning default expression.")
        return 0.0
