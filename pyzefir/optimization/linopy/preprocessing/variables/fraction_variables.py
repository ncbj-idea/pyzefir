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
    """
    Class representing the fraction variables for local balancing stacks.

    This class encapsulates the fraction variables that represent the proportion of
    local balancing stacks (LBS) allocated to each aggregated consumer
    over different years. These variables can be binary or continuous based on the
    specified parameters.
    """

    def __init__(self, model: Model, indices: Indices, binary_fraction: bool = False):
        """
        Initializes a new instance of the class.

        Args:
            - model (Model): The optimization model to which the variables will be added.
            - indices (Indices): The indices used for mapping aggregated consumers,
              local balancing stacks, and years.
            - binary_fraction (bool, optional): A flag indicating whether the fraction
              variables should be binary. Defaults to False (continuous variables).
        """
        if binary_fraction:
            self.fraction = model.add_variables(
                coords=[indices.AGGR.ii, indices.LBS.ii, indices.Y.ii],
                dims=["aggr", "lbs", "year"],
                name="FRACTION",
                binary=binary_fraction,
            )
        else:
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
