from gurobipy import MLinExpr, quicksum

from pyzefir.optimization.gurobi.objective_builder import ObjectiveBuilder


class VarCostObjectiveBuilder(ObjectiveBuilder):
    def build_expression(self) -> MLinExpr:
        return quicksum(
            self.generator_var_cost(gen_idx)
            for gen_idx in self.indices.GEN.ord
            if self.parameters.gen.fuel[gen_idx] is not None
        ).sum()

    def generator_var_cost(self, gen_idx: int) -> MLinExpr | float:
        fuel_idx = self.parameters.gen.fuel[gen_idx]
        hourly_scale = self.parameters.scenario_parameters.hourly_scale
        cost = self.parameters.fuel.unit_cost[fuel_idx]
        fuel_consumption = self.expr.fuel_consumption(fuel_idx, gen_idx, hourly_scale)

        return fuel_consumption * cost
