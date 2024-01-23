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
from pyzefir.optimization.gurobi.preprocessing.variables import VariableGroup
from pyzefir.optimization.opt_config import OptConfig


class BusVariables(VariableGroup):
    """Bus variables"""

    def __init__(self, model: Model, indices: Indices, opt_config: OptConfig) -> None:
        self.bus_ens = model.addMVar(
            shape=(len(indices.BUS), len(indices.H), len(indices.Y)),
            name="BUS_ENS",
            ub=0 if not opt_config.ens else None,
        )
        """ bus variables"""
