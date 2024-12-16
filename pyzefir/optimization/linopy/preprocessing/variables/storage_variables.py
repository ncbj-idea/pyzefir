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
from linopy import Model, Variable

from pyzefir.model.network import Network
from pyzefir.model.utils import AllowedStorageGenerationLoadMethods
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.variables import VariableGroup
from pyzefir.optimization.linopy.preprocessing.variables.demand_chunks import (
    create_dch_vars,
)
from pyzefir.optimization.linopy.preprocessing.variables.utils import add_h_y_variable


class StorageVariables(VariableGroup):
    """
    Class representing the storage variables.

    This class encapsulates the variables associated with energy storage units in the
    network, including generation, load, state of charge, and capacity metrics. These
    variables are critical for modeling the behavior and performance of storage systems
    in energy management.
    """

    def __init__(self, model: Model, indices: Indices, network: Network) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - model (Model): The optimization model to which the storage variables will be added.
            - indices (Indices): The indices used for mapping storage parameters across
              different time periods and storage units.
            - network (Network): The network representation that includes demand chunks and
              storage elements.
        """
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
        self.gen_dch = create_dch_vars(
            model=model,
            demand_chunks=network.demand_chunks,
            energy_sources=network.storages,
            indices=indices,
            energy_source_ii=indices.STOR,
            var_name="ST_DEM_CH",
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

        self.milp_bin = self._create_storage_binary_variables(
            model=model,
            indices=indices,
            network=network,
            method=AllowedStorageGenerationLoadMethods.milp,
        )
        """ binary variables indicating if a storage unit is active in a given hour and year """

    def _create_storage_binary_variables(
        self,
        model: Model,
        indices: Indices,
        network: Network,
        method: AllowedStorageGenerationLoadMethods,
    ) -> dict[tuple[int, int], Variable]:
        """
        Creates binary variables for storage units in the model based on the specified generation/load method.

        This method iterates through storage units in the provided indices, checks their corresponding
        generation/load method, and if it matches the specified method, it creates a binary decision variable
        for that storage. The variable is added to the model and stored in a dictionary with a key composed
        of the storage index and the storage type index.

        Args:
            model (Model): The MILP model to which binary variables will be added.
            indices (Indices): Contains mappings between storage indices and types.
            network (Network): The network object containing information about the storages and their types.
            method (AllowedStorageGenerationLoadMethods): The method type (e.g., MILP) that determines whether
                                                        binary variables should be created for the storage.

        Returns:
            dict[tuple[int, int], Variable]: A dictionary where the key is a tuple of the storage index and
                                            the storage type index, and the value is the corresponding binary
                                            variable added to the model.
        """
        binary_variables_dict: dict[tuple[int, int], Variable] = {}
        for storage_idx, storage_name in indices.STOR.mapping.items():
            energy_source_type = network.storages[str(storage_name)].energy_source_type
            storage_type = network.storage_types[energy_source_type]
            if storage_type.generation_load_method == method:
                storage_type_idx = indices.TSTOR.inverse[storage_type.name]
                binary_variable = add_h_y_variable(
                    model=model,
                    indices=indices,
                    var_name=f"{storage_name}_{method}_bin",
                    use_binary=True,
                )
                binary_variables_dict[(storage_idx, storage_type_idx)] = binary_variable
        return binary_variables_dict
