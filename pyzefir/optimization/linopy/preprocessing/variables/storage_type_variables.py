# PyZefir
# Copyright (C) 2024 Narodowe Centrum Badań Jądrowych
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

import numpy as np
import xarray as xr
from linopy import Model

from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.variables import VariableGroup


class StorageTypeVariables(VariableGroup):
    """
    Class representing the storage type variables.

    This class encapsulates the variables related to different storage types within the
    energy network. It includes capacity variables, capacity increases, and decreases
    over time, which are essential for modeling the operational flexibility and dynamics
    of storage units in the energy system.
    """

    def __init__(self, model: Model, indices: Indices) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - model (Model): The optimization model to which the storage type variables will be added.
            - indices (Indices): The indices used for mapping storage type parameters across
              different aggregated indices and years.
        """
        indexes = [
            (aggr_idx, aggr_stor_type_idx, year_idx)
            for aggr_idx, aggr_stor_type_idxs in indices.aggr_tstor_map.items()
            for aggr_stor_type_idx in aggr_stor_type_idxs
            for year_idx in indices.Y.ord
        ]

        self.tcap = model.add_variables(
            lower=xr.DataArray(
                np.full(len(indexes), 0),
                dims=["index"],
                coords=dict(index=np.array(indexes, dtype="i,i,i")),
                name="tcap",
            ),
            name="STOR_TYPE_CAP",
        )
        """ storage type capacity """

        self.tcap_plus = model.add_variables(
            lower=xr.DataArray(
                np.full(len(indexes), 0),
                dims=["index"],
                coords=dict(index=np.array(indexes, dtype="i,i,i")),
                name="tcap_plus",
            ),
            name="STOR_TYPE_CAP_PLUS",
        )
        """ storage type capacity increase """

        self.tcap_minus = model.add_variables(
            lower=xr.DataArray(
                np.full(len(indexes) * len(indices.Y), 0),
                dims=["index"],
                coords=dict(
                    index=np.array(
                        [
                            index + (year,)
                            for index, year in product(indexes, indices.Y.ii)
                        ],
                        dtype="i,i,i,i",
                    ),
                ),
                name="tcap_minus",
            ),
            name="STOR_TYPE_CAP_MINUS",
        )
        """ storage type capacity decrease """

        self.tcap_base_minus = model.add_variables(
            lower=xr.DataArray(
                np.full(len(indexes), 0),
                dims=["index"],
                coords=dict(index=np.array(indexes, dtype="i,i,i")),
                name="tcap_base_minus",
            ),
            name="STOR_TYPE_CAP_BASE_MINUS",
        )
        """ storage type base capacity decrease """
