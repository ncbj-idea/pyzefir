import numpy as np

from pyzefir.optimization.gurobi.constraints_builder.builder import (
    PartialConstraintsBuilder,
)


class LineFlowConstraintsBuilder(PartialConstraintsBuilder):
    def build_constraints(self) -> None:
        self.max_flow_constraints()

    def max_flow_constraints(self) -> None:
        for line_idx, line_name in self.indices.LINE.mapping.items():
            flow = self.variables.line.flow[line_idx, :, :]
            max_capacity = self.parameters.line.cap[line_idx]
            if not np.isnan(max_capacity):
                self.model.addConstr(
                    flow <= max_capacity,
                    name=f"{line_name}_LINE_FLOW_UPPER_BOUND_CONSTRAINT",
                )
