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

import numpy as np
from gurobipy import MLinExpr, quicksum

from pyzefir.optimization.gurobi.objective_builder import ObjectiveBuilder


class TransmissionFeeObjectiveBuilder(ObjectiveBuilder):
    def build_expression(self) -> MLinExpr | float:
        if len(self.parameters.line.tf) == 0:
            return 0.0

        return (
            quicksum(
                self.line_flow_cost(line_idx, tf_idx)
                for line_idx, tf_idx in self.parameters.line.tf.items()
            ).sum()
            * self.parameters.scenario_parameters.hourly_scale
        )

    def line_flow_cost(self, line_idx: int, tf_idx: int) -> MLinExpr:
        line_flow_cost = np.tile(
            self.parameters.tf.fee[tf_idx], (self.variables.line.flow.shape[2], 1)
        ).transpose()
        line_flow = self.variables.line.flow[line_idx, :, :]

        return line_flow * line_flow_cost
