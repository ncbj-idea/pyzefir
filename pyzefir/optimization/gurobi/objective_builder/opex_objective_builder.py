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
