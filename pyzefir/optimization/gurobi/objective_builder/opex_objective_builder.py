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
from gurobipy import MLinExpr, MVar

from pyzefir.optimization.gurobi.objective_builder import ObjectiveBuilder
from pyzefir.optimization.gurobi.preprocessing.indices import IndexingSet


class OpexObjectiveBuilder(ObjectiveBuilder):
    def build_expression(self) -> MLinExpr:
        return self.generator_opex() + self.storage_opex()

    def generator_opex(self) -> MLinExpr:
        return self._opex_expression(
            self.indices.GEN,
            self.variables.gen.cap,
            self.parameters.gen.tgen,
            self.parameters.tgen.opex,
        )

    def storage_opex(self) -> MLinExpr:
        return self._opex_expression(
            self.indices.STOR,
            self.variables.stor.cap,
            self.parameters.stor.tstor,
            self.parameters.tstor.opex,
        )

    @staticmethod
    def _opex_expression(
        unit_index: IndexingSet, cap: MVar, type_gen: dict, opex: np.ndarray
    ) -> MLinExpr | float:
        result = 0.0
        for u_idx in unit_index.ord:
            result += (opex[type_gen[u_idx]] * cap[u_idx, :]).sum()
        return result
