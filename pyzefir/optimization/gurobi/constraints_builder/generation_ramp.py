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
