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
    """
    Class representing the line variables.

    This class encapsulates the variables associated with transmission lines in the
    energy network, including flow variables over time. These variables are crucial for
    modeling the energy transfer capacity and constraints of the lines connecting different
    nodes in the network.
    """

    def __init__(self, model: Model, indices: Indices) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - model (Model): The optimization model to which the line variables will be added.
            - indices (Indices): The indices used for mapping line, hour, and year parameters.
        """
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
