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

from pyzefir.model.network import Network
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.variables import VariableGroup
from pyzefir.optimization.linopy.preprocessing.variables.utils import add_h_y_variable
from pyzefir.optimization.opt_config import OptConfig


class BusVariables(VariableGroup):
    """
    Class representing the bus variables.

    This class encapsulates the bus-related variables in the model, including
    energy not served (ENS) variables and shift variables for demand-side response (DSR) buses.
    """

    def __init__(
        self,
        model: Model,
        network: Network,
        indices: Indices,
        opt_config: OptConfig,
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - model (Model): The optimization model to which these variables belong.
            - network (Network): The network representation of the system.
            - indices (Indices): The indices for accessing various components in the model.
            - opt_config (OptConfig): Configuration parameters for the optimization.
        """
        self.bus_ens = model.add_variables(
            lower=0,
            upper=xr.DataArray(
                np.full(
                    (len(indices.BUS), len(indices.H), len(indices.Y)),
                    np.inf if not np.isnan(opt_config.ens) else 0,
                ),
                dims=["bus", "hour", "year"],
                coords=[indices.BUS.ii, indices.H.ii, indices.Y.ii],
                name="bus_ens",
            ),
            name="BUS_ENS",
        )
        """ ens variable """
        bus_with_dsr_indices = self.buses_with_dsr(network, indices)
        self.shift_minus = self.create_shift_variable(
            model, indices, bus_with_dsr_indices, var_name="SHIFT_MINUS"
        )
        """ down shift for dsr """
        self.shift_plus = self.create_shift_variable(
            model, indices, bus_with_dsr_indices, var_name="SHIFT_PLUS"
        )
        """ up shift for dsr """

    @staticmethod
    def buses_with_dsr(network: Network, indices: Indices) -> list[int]:
        """
        Returns all indices of buses with dsr type defined.

        Args:
            - network (Network): network representation of the model
            - indices (Indices): indices of buses of interest

        Returns:
            - list[int]: indices of buses with dsr
        """
        return [
            indices.BUS.inverse[bus.name]
            for bus in network.buses.values()
            if bus.dsr_type is not None
        ]

    @staticmethod
    def create_shift_variable(
        model: Model, indices: Indices, buses_indices: list[int], var_name: str
    ) -> dict[int, Variable]:
        """
        Creates shift variable for buses.

        Args:
            - model (Model): model
            - indices (Indices): specified indices of variable
            - buses_indices (list[int]): indices of buses
            - var_name (str): name of the variable

        Returns:
            - dict[int, Variable]: dict with bus index as keys and variables as values
        """
        return {
            bus_idx: add_h_y_variable(
                model, indices, var_name=f"{var_name}_{indices.BUS.mapping[bus_idx]}"
            )
            for bus_idx in buses_indices
        }
