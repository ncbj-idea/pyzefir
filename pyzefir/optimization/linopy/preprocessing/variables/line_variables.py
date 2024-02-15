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

import numpy as np
import xarray as xr
from linopy import Model

from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.variables import VariableGroup


class LineVariables(VariableGroup):
    """Line variables"""

    def __init__(self, model: Model, indices: Indices):
        self.flow = model.add_variables(
            lower=xr.DataArray(
                np.full((len(indices.LINE), len(indices.H), len(indices.Y)), 0),
                dims=["line", "hour", "year"],
                coords=[indices.LINE.ii, indices.H.ii, indices.Y.ii],
                name="flow",
            ),
            name="L_FLOW",
        )
        """ line flow """
