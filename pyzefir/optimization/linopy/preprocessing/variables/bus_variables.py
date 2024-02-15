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
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.linopy.preprocessing.variables import VariableGroup
from pyzefir.optimization.opt_config import OptConfig


class BusVariables(VariableGroup):
    """Bus variables"""

    def __init__(
        self,
        model: Model,
        indices: Indices,
        opt_config: OptConfig,
        parameters: OptimizationParameters | None,
    ) -> None:
        self.bus_ens = model.add_variables(
            lower=0,
            upper=xr.DataArray(
                np.full(
                    (len(indices.BUS), len(indices.H), len(indices.Y)),
                    np.inf if opt_config.ens else 0,
                ),
                dims=["bus", "hour", "year"],
                coords=[indices.BUS.ii, indices.H.ii, indices.Y.ii],
                name="bus_ens",
            ),
            name="BUS_ENS",
        )
        n_dsr = len(parameters.bus.dsr_type) if parameters is not None else 0
        bus_dsr_index = (
            [indices.BUS.mapping[key] for key in parameters.bus.dsr_type.keys()]
            if parameters
            else []
        )
        """ bus variables"""
        self.shift_minus = model.add_variables(
            lower=xr.DataArray(
                np.full(
                    (n_dsr, len(indices.H), len(indices.Y)),
                    0,
                ),
                dims=["bus", "hour", "year"],
                coords=[bus_dsr_index, indices.H.ii, indices.Y.ii],
                name="SHIFT_MINUS",
            ),
        )
        """ down shift for dsr """
        self.shift_plus = model.add_variables(
            lower=xr.DataArray(
                np.full(
                    (n_dsr, len(indices.H), len(indices.Y)),
                    0,
                ),
                dims=["bus", "hour", "year"],
                coords=[bus_dsr_index, indices.H.ii, indices.Y.ii],
                name="SHIFT_PLUS",
            ),
        )
        """ up shift for dsr """
