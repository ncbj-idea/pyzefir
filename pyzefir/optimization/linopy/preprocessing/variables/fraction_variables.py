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


class FractionVariables(VariableGroup):
    """Fraction variables"""

    def __init__(self, model: Model, indices: Indices, binary_fraction: bool = False):
        self.fraction = model.add_variables(
            lower=xr.DataArray(
                np.full((len(indices.AGGR), len(indices.LBS), len(indices.Y)), 0),
                dims=["aggr", "lbs", "year"],
                coords=[indices.AGGR.ii, indices.LBS.ii, indices.Y.ii],
                name="fraction",
            ),
            name="FRACTION",
            binary=binary_fraction,
        )
        """ fraction of local balancing stack in a given aggregated consumer """
