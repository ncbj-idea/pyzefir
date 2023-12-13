import numpy as np

from pyzefir.optimization.gurobi.constraints_builder.builder import (
    PartialConstraintsBuilder,
)


class RampConstraintsBuilder(PartialConstraintsBuilder):
    def build_constraints(self) -> None:
        self.ramp_up_constraint()

    def ramp_up_constraint(self) -> None:
        for gen_idx, gen_name in self.indices.GEN.mapping.items():
            t_idx = self.parameters.gen.tgen[gen_idx]
            ramp = self.parameters.tgen.ramp[t_idx]
            if not np.isnan(ramp):
                gen = self.variables.gen.gen
                cap = self.variables.gen.cap
                self.model.addConstr(
                    gen[gen_idx, 1:, :] - gen[gen_idx, :-1, :]
                    <= cap[gen_idx, :] * ramp,
                    name=f"{gen_name}_RAMP_PLUS_CONSTRAINT",
                )
                self.model.addConstr(
                    -gen[gen_idx, 1:, :] + gen[gen_idx, :-1, :]
                    <= cap[gen_idx, :] * ramp,
                    name=f"{gen_name}_RAMP_PLUS_CONSTRAINT",
                )
