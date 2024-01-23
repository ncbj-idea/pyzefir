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
