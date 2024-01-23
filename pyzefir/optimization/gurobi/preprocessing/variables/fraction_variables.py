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

from gurobipy import GRB, Model

from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.variables import VariableGroup


class FractionVariables(VariableGroup):
    """Fraction variables"""

    def __init__(
        self, grb_model: Model, indices: Indices, binary_fraction: bool = False
    ):
        v_type = GRB.BINARY if binary_fraction else GRB.CONTINUOUS
        self.fraction = grb_model.addMVar(
            (len(indices.AGGR), len(indices.LBS), len(indices.Y)),
            name="FRACTION",
            vtype=v_type,
        )
        """ fraction of local balancing stack in a given aggregated consumer """
