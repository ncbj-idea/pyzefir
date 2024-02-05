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

from gurobipy import MLinExpr, quicksum

from pyzefir.optimization.gurobi.objective_builder import ObjectiveBuilder


class DsrPenaltyObjectiveBuilder(ObjectiveBuilder):
    def build_expression(self) -> MLinExpr | float:
        if len(self.parameters.bus.dsr_type):
            shift_minus = self.variables.bus.shift_minus
            penalization = self.parameters.dsr.penalization
            bus_parameters = self.parameters.bus
            return quicksum(
                shift_minus[bus_idx, :, :].sum() * penalization[dsr_idx]
                for bus_idx, dsr_idx in bus_parameters.dsr_type.items()
            )
        else:
            return 0.0
