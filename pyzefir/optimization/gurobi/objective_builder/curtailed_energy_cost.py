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


class CurtailedEnergyCostObjectiveBuilder(ObjectiveBuilder):
    def build_expression(self) -> MLinExpr:
        curtailment_cost = self.parameters.tgen.energy_curtailment_cost
        gen_ett = {
            k: {self.indices.ET.inverse[et] for et in v}
            for k, v in self.parameters.gen.ett.items()
        }
        gen_idx_to_tgen_idx = {
            gen_idx: tgen_idx
            for gen_idx, tgen_idx in self.parameters.gen.tgen.items()
            if tgen_idx in curtailment_cost
        }
        return quicksum(
            self.variables.gen.dump_et[gen_idx, et, :, :].sum(axis=0)
            @ curtailment_cost[tgen_idx]
            for gen_idx, tgen_idx in gen_idx_to_tgen_idx.items()
            for et in gen_ett[gen_idx]
        )
