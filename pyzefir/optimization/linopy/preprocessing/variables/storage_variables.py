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


class StorageVariables(VariableGroup):
    """Storage variables"""

    def __init__(self, model: Model, indices: Indices) -> None:
        self.gen = model.add_variables(
            lower=xr.DataArray(
                np.full((len(indices.STOR), len(indices.H), len(indices.Y)), 0),
                dims=["stor", "hour", "year"],
                coords=[indices.STOR.ii, indices.H.ii, indices.Y.ii],
                name="gen",
            ),
            name="S_GEN",
        )
        """ generation """

        self.gen_dch = model.add_variables(
            lower=xr.DataArray(
                np.full(
                    (
                        len(indices.DEMCH),
                        len(indices.STOR),
                        len(indices.H),
                        len(indices.Y),
                    ),
                    0,
                ),
                dims=["demch", "stor", "hour", "year"],
                coords=[indices.DEMCH.ii, indices.STOR.ii, indices.H.ii, indices.Y.ii],
                name="gen_dch",
            ),
            name="S_GEN_DCH",
        )
        """ generation to cover demand chunks """

        self.load = model.add_variables(
            lower=xr.DataArray(
                np.full((len(indices.STOR), len(indices.H), len(indices.Y)), 0),
                dims=["stor", "hour", "year"],
                coords=[indices.STOR.ii, indices.H.ii, indices.Y.ii],
                name="load",
            ),
            name="S_LOAD",
        )
        """ load """

        self.soc = model.add_variables(
            lower=xr.DataArray(
                np.full((len(indices.STOR), len(indices.H), len(indices.Y)), 0),
                dims=["stor", "hour", "year"],
                coords=[indices.STOR.ii, indices.H.ii, indices.Y.ii],
                name="soc",
            ),
            name="S_SOC",
        )
        """ state of charge """

        self.cap = model.add_variables(
            lower=xr.DataArray(
                np.full((len(indices.STOR), len(indices.Y)), 0),
                dims=["stor", "year"],
                coords=[indices.STOR.ii, indices.Y.ii],
                name="cap",
            ),
            name="S_CAP",
        )
        """ capacity """

        non_aggr_stor_idx = set(indices.STOR.mapping.keys()) - {
            aggr_stor_idx
            for aggr_stor_idxs in indices.aggr_stor_map.values()
            for aggr_stor_idx in aggr_stor_idxs
        }
        indexes = list(product(non_aggr_stor_idx, indices.Y.ord))

        self.cap_plus = model.add_variables(
            lower=xr.DataArray(
                np.full(len(indexes), 0),
                dims=["index"],
                coords=dict(index=np.array(indexes, dtype="i,i")),
                name="cap_plus",
            ),
            name="S_CAP_PLUS",
        )
        """ capacity increase """

        self.cap_minus = model.add_variables(
            lower=xr.DataArray(
                np.full(len(indexes) * len(indices.Y), 0),
                dims=["index"],
                coords=dict(
                    index=np.array(
                        list(product(non_aggr_stor_idx, indices.Y.ord, indices.Y.ord)),
                        dtype="i,i,i",
                    )
                ),
                name="cap_minus",
            ),
            name="S_CAP_MINUS",
        )
        """ capacity decrease """

        self.cap_base_minus = model.add_variables(
            lower=xr.DataArray(
                np.full(len(indexes), 0),
                dims=["index"],
                coords=dict(index=np.array(indexes, dtype="i,i")),
                name="cap_base_minus",
            ),
            name="S_CAP_BASE_MINUS",
        )
        """ base capacity decrease """
