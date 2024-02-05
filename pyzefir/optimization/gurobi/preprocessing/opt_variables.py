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

from gurobipy import Model

from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.gurobi.preprocessing.variables.bus_variables import (
    BusVariables,
)
from pyzefir.optimization.gurobi.preprocessing.variables.fraction_variables import (
    FractionVariables,
)
from pyzefir.optimization.gurobi.preprocessing.variables.generator_type_variables import (
    GeneratorTypeVariables,
)
from pyzefir.optimization.gurobi.preprocessing.variables.generator_variables import (
    GeneratorVariables,
)
from pyzefir.optimization.gurobi.preprocessing.variables.line_variables import (
    LineVariables,
)
from pyzefir.optimization.gurobi.preprocessing.variables.storage_type_variables import (
    StorageTypeVariables,
)
from pyzefir.optimization.gurobi.preprocessing.variables.storage_variables import (
    StorageVariables,
)
from pyzefir.optimization.opt_config import OptConfig


class OptimizationVariables:
    """
    All optimization variables.
    """

    def __init__(
        self,
        grb_model: Model,
        indices: Indices,
        opt_config: OptConfig,
        parameters: OptimizationParameters | None = None,
    ) -> None:
        self.gen = GeneratorVariables(grb_model, indices)
        """ generators variables """
        self.stor = StorageVariables(grb_model, indices)
        """ storage variables """
        self.line = LineVariables(grb_model, indices)
        """ line variables """
        self.frac = FractionVariables(grb_model, indices)
        """ fraction variables """
        self.bus = BusVariables(grb_model, indices, opt_config, parameters)
        """ bus variables """
        self.tgen = GeneratorTypeVariables(grb_model, indices)
        """ generator type variables """
        self.tstor = StorageTypeVariables(grb_model, indices)
        """ storage type variables """
