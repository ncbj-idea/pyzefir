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
    def build_expression(self) -> LinearExpression | float:
        _logger.info("Building DSR penalty objective...")
        if self.parameters.bus.dsr_type:
            shift_minus = self.variables.bus.shift_minus
            penalization = self.parameters.dsr.penalization
            bus_parameters = self.parameters.bus
            result: LinearExpression = 0.0
            for bus_idx, dsr_idx in bus_parameters.dsr_type.items():
                result += (
                    shift_minus.isel(bus=bus_idx) * self.indices.years_aggregation_array
                ).sum() * penalization[dsr_idx]
            _logger.info("DSR penalty objective: Done")
            return result.sum()
        _logger.warning("No specified DSR type, returning default expression.")
        return 0.0
