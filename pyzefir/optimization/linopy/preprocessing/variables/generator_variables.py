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
from collections import defaultdict
from itertools import product
from typing import Iterable

import numpy as np
import xarray as xr
from linopy import Model, Variable

from pyzefir.model.network import Network
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.variables import VariableGroup
from pyzefir.optimization.linopy.preprocessing.variables.demand_chunks import (
    create_dch_vars,
)
from pyzefir.optimization.linopy.preprocessing.variables.utils import add_h_y_variable


class GeneratorVariables(VariableGroup):
    """
    Class representing the generator variables.

    This class encapsulates the variables related to generators in the energy
    network, including generation amounts, capacities, and changes in capacities
    over time. These variables are critical for modeling the performance and
    constraints of various generator types in response to energy demands.
    """

    def __init__(
        self,
        model: Model,
        indices: Indices,
        network: Network,
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - model (Model): The optimization model to which the variables will be added.
            - indices (Indices): The indices used for mapping generator and year parameters.
            - network (Network): The network representation that includes generator details
              and relationships with demand chunks.
        """
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
        self.gen_et = self._create_generation_variable(
            network, indices, model, "GEN_ET"
        )

        self.gen_reserve_et = self._create_gen_et_reserve(network, indices, model)
        """ frozen part of generation associated with power reserve"""

        """ generation to cover demand chunks """
        self.gen_dch = create_dch_vars(
            model=model,
            demand_chunks=network.demand_chunks,
            energy_sources=network.generators,
            indices=indices,
            energy_source_ii=indices.GEN,
            var_name="G_DEM_CH",
        )
        """ energy_type -> dump[energy_type] """
        disabled_dump_idxs = {
            gen_idx
            for gen_idx in network.generators
            if network.generator_types[
                network.generators[gen_idx].energy_source_type
            ].disable_dump_energy
        }
        self.dump_et = self._create_generation_variable(
            network, indices, model, "DUMP_ET", disabled_dump_idxs
        )
        """ capacity """
        self.cap = model.add_variables(
            lower=xr.DataArray(
                np.full((len(indices.GEN), len(indices.Y)), 0),
                dims=["gen", "year"],
                coords=[indices.GEN.ii, indices.Y.ii],
            ),
            name="G_CAP",
        )
        non_aggr_gen_idx = set(indices.GEN.mapping.keys()) - {
            aggr_gen_idx
            for aggr_gen_idxs in indices.aggr_gen_map.values()
            for aggr_gen_idx in aggr_gen_idxs
        }
        indexes = list(product(non_aggr_gen_idx, indices.Y.ord))
        """ capacity increase """
        self.cap_plus = model.add_variables(
            lower=xr.DataArray(
                np.full(len(indexes), 0),
                dims=["index"],
                coords=dict(index=np.array(indexes, dtype="i,i")),
            ),
            name="G_CAP_PLUS",
        )
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
            ),
            name="G_CAP_MINUS",
        )
        self.cap_base_minus = model.add_variables(
            lower=xr.DataArray(
                np.full(len(indexes), 0),
                dims=["index"],
                coords=dict(index=np.array(indexes, dtype="i,i")),
            ),
            name="G_CAP_BASE_MINUS",
        )
        """ base capacity decrease """

    @staticmethod
    def _create_generation_variable(
        network: Network,
        indices: Indices,
        model: Model,
        var_name: str,
        exception: Iterable | None = None,
    ) -> dict[int, dict[str, Variable]]:
        """
        Create generation variable. (gen_idx -> energy_type -> Var[h, y])

        Args:
            - network (Network): network representation of the model
            - indices (Indices): indices of the new variable
            - model (Model): model
            - var_name (str): name of the variable

        Returns:
            - dict[int, dict[str, Variable]]: dict of created variables
        """
        result: defaultdict[int, dict[str, Variable]] = defaultdict(dict)
        for gen_obj in network.generators.values():
            if exception is None or gen_obj.name not in exception:
                gen_idx = indices.GEN.inverse[gen_obj.name]
                for et in network.generator_types[
                    gen_obj.energy_source_type
                ].energy_types:
                    result[gen_idx][et] = add_h_y_variable(
                        model, indices, var_name=f"{var_name}_{gen_obj.name}_{et}"
                    )
        return dict(result)

    @staticmethod
    def _create_gen_et_reserve(
        network: Network,
        indices: Indices,
        model: Model,
    ) -> dict[int, dict[int, dict[str, Variable]]]:
        """
        Create generation variable for reserves, which maps tag_idx to gen_idx to energy_type to Var[h, y].
        The reason for this is the fact, that every power reserve "frozen generation" is determined by tag idx
        generator idx and energy type.

        Args:
            - network (Network): network representation of the model
            - indices (Indices): indices of the new variable
            - model (Model): model

        Returns:
            - dict[int, dict[int, dict[str, Variable]]]: dict of created variables
        """
        result: dict[int, dict[int, dict[str, Variable]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for gen_name, gen_obj in network.generators.items():
            gen_idx = indices.GEN.inverse[gen_name]
            for et, power_reserve_data in network.constants.power_reserves.items():
                for tag in power_reserve_data:
                    if tag in gen_obj.tags:
                        result[indices.TAGS.inverse[tag]][gen_idx][et] = (
                            add_h_y_variable(
                                model,
                                indices,
                                var_name=f"GEN_RESERVE_ET_{tag}_{gen_obj.name}_{et}",
                            )
                        )
        return result
