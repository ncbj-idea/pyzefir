import abc

from gurobipy import Model

from pyzefir.optimization.gurobi.expression_handler import ExpressionHandler
from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.gurobi.preprocessing.opt_variables import (
    OptimizationVariables,
)


class PartialConstraintsBuilder(metaclass=abc.ABCMeta):
    """
    An abstract class to represent build of some constraints set
    """

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

    @abc.abstractmethod
    def build_constraints(self) -> None:
        """
        Creating optimization constraints.
        """
        raise NotImplementedError
