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
from linopy import Model, Variable

from pyzefir.optimization.linopy.preprocessing.indices import Indices


def add_h_y_variable(
    model: Model, indices: Indices, var_name: str, use_binary: bool = False
) -> Variable:
    """
    Add Var[hour, year] to the model.

    Args:
        - model (Model): The optimization model to which the variable will be added.
        - indices (Indices): The indices used for mapping the variable across hours and years.
        - var_name (str): The name of the variable to be created.
        - use_binary (bool): If to use binary. Defaults to False.

    Returns:
        - Variable: The newly created variable with dimensions for hours and years.
    """
    if use_binary:
        return model.add_variables(
            coords=[indices.H.ii, indices.Y.ii],
            dims=["hour", "year"],
            name=var_name,
            binary=True,
        )
    return model.add_variables(
        lower=xr.DataArray(
            np.full((len(indices.H), len(indices.Y)), 0),
            dims=["hour", "year"],
            coords=[indices.H.ii, indices.Y.ii],
        ),
        name=var_name,
    )
