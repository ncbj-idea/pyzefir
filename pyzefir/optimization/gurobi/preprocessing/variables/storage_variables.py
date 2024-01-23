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

from itertools import product

from gurobipy import Model

from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.variables import VariableGroup


class StorageVariables(VariableGroup):
    """Storage variables"""

    def __init__(self, grb_model: Model, indices: Indices) -> None:
        self.gen = grb_model.addMVar(
            (len(indices.STOR), len(indices.H), len(indices.Y)), name="ST_GEN"
        )
        """ generation """
        self.gen_dch = grb_model.addMVar(
            shape=(
                len(indices.DEMCH),
                len(indices.STOR),
                len(indices.H),
                len(indices.Y),
            ),
            name="ST_GEN_DCH",
        )
        """ generation to cover demand chunks """
        self.load = grb_model.addMVar(
            (len(indices.STOR), len(indices.H), len(indices.Y)), name="ST_LOAD"
        )
        """ load """
        self.soc = grb_model.addMVar(
            (len(indices.STOR), len(indices.H), len(indices.Y)), name="ST_SOC"
        )
        """ state of charge """
        self.cap = grb_model.addMVar((len(indices.STOR), len(indices.Y)), name="ST_CAP")
        """ capacity """
        non_aggr_stor_idx = set(indices.STOR.mapping.keys()) - {
            aggr_stor_idx
            for aggr_stor_idxs in indices.aggr_stor_map.values()
            for aggr_stor_idx in aggr_stor_idxs
        }
        indexes = list(product(non_aggr_stor_idx, indices.Y.ord))
        self.cap_plus = grb_model.addVars(indexes, name="ST_CAP_PLUS")
        """ capacity increase """
        self.cap_minus = grb_model.addVars(
            list(product(non_aggr_stor_idx, indices.Y.ord, indices.Y.ord)),
            name="ST_CAP_MINUS",
        )
        """ capacity decrease """
        self.cap_base_minus = grb_model.addVars(indexes, name="ST_CAP_BASE_MINUS")
        """ base capacity decrease """
