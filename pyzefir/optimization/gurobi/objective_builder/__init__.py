from abc import abstractmethod

from gurobipy import MLinExpr, Model

from pyzefir.optimization.gurobi.expression_handler import ExpressionHandler
from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.gurobi.preprocessing.opt_variables import (
    OptimizationVariables,
)


class ObjectiveBuilder:
    def __init__(
        self,
        indices: Indices,
        parameters: OptimizationParameters,
        variables: OptimizationVariables,
        model: Model,
    ) -> None:
        self.indices = indices
        self.parameters = parameters
        self.variables = variables
        self.model = model
        self.expr = ExpressionHandler(indices, variables, parameters)

    @abstractmethod
    def build_expression(self) -> MLinExpr:
        pass
