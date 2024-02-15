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


class GeneratorVariables(VariableGroup):
    """Generator variables"""

    def __init__(self, model: Model, indices: Indices) -> None:
        self.gen = model.add_variables(
            lower=xr.DataArray(
                np.full((len(indices.GEN), len(indices.H), len(indices.Y)), 0),
                dims=["gen", "hour", "year"],
                coords=[indices.GEN.ii, indices.H.ii, indices.Y.ii],
                name="gen",
            ),
            name="G_GEN",
        )
        """ generation """

        self.gen_et = model.add_variables(
            lower=xr.DataArray(
                np.full(
                    (len(indices.GEN), len(indices.ET), len(indices.H), len(indices.Y)),
                    0,
                ),
                dims=["gen", "et", "hour", "year"],
                coords=[indices.GEN.ii, indices.ET.ii, indices.H.ii, indices.Y.ii],
                name="gen_et",
            ),
            name="G_GEN_ET",
        )
        """ energy_type -> generation[energy_type] """

        self.gen_dch = model.add_variables(
            lower=xr.DataArray(
                np.full(
                    (
                        len(indices.ET),
                        len(indices.DEMCH),
                        len(indices.GEN),
                        len(indices.H),
                        len(indices.Y),
                    ),
                    0,
                ),
                dims=["et", "demch", "gen", "hour", "year"],
                coords=[
                    indices.ET.ii,
                    indices.DEMCH.ii,
                    indices.GEN.ii,
                    indices.H.ii,
                    indices.Y.ii,
                ],
                name="gen_dch",
            ),
            name="G_GEN_DCH",
        )
        """ generation to cover demand chunks """
        self.dump = model.add_variables(
            lower=xr.DataArray(
                np.full((len(indices.GEN), len(indices.H), len(indices.Y)), 0),
                dims=["gen", "hour", "year"],
                coords=[indices.GEN.ii, indices.H.ii, indices.Y.ii],
                name="dump",
            ),
            name="G_DUMP",
        )
        """ dump energy """
        self.dump_et = model.add_variables(
            lower=xr.DataArray(
                np.full(
                    (len(indices.GEN), len(indices.ET), len(indices.H), len(indices.Y)),
                    0,
                ),
                dims=["gen", "et", "hour", "year"],
                coords=[indices.GEN.ii, indices.ET.ii, indices.H.ii, indices.Y.ii],
                name="dump_et",
            ),
            name="G_DUMP_ET",
        )
        """ energy_type -> dump[energy_type] """
        self.cap = model.add_variables(
            lower=xr.DataArray(
                np.full((len(indices.GEN), len(indices.Y)), 0),
                dims=["gen", "year"],
                coords=[indices.GEN.ii, indices.Y.ii],
                name="cap",
            ),
            name="G_CAP",
        )
        """ capacity """
        non_aggr_gen_idx = set(indices.GEN.mapping.keys()) - {
            aggr_gen_idx
            for aggr_gen_idxs in indices.aggr_gen_map.values()
            for aggr_gen_idx in aggr_gen_idxs
        }
        indexes = list(product(non_aggr_gen_idx, indices.Y.ord))
        self.cap_plus = model.add_variables(
            lower=xr.DataArray(
                np.full(len(indexes), 0),
                dims=["index"],
                coords=dict(index=np.array(indexes, dtype="i,i")),
                name="cap_plus",
            ),
            name="G_CAP_PLUS",
        )
        """ capacity increase """
        self.cap_minus = model.add_variables(
            lower=xr.DataArray(
                np.full(len(indexes) * len(indices.Y), 0),
                dims=["index"],
                coords=dict(
                    index=np.array(
                        [
                            index + (year,)
                            for index, year in product(indexes, indices.Y.ii)
                        ],
                        dtype="i,i,i",
                    ),
                ),
                name="cap_minus",
            ),
            name="G_CAP_MINUS",
        )
        """ capacity decrease """
        self.cap_base_minus = model.add_variables(
            lower=xr.DataArray(
                np.full(len(indexes), 0),
                dims=["index"],
                coords=dict(index=np.array(indexes, dtype="i,i")),
                name="cap_base_minus",
            ),
            name="G_CAP_BASE_MINUS",
        )
        """ base capacity decrease """
