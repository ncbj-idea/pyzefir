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
