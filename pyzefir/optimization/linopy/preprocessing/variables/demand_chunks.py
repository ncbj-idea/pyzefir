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

from linopy import Model, Variable

from pyzefir.model.network import NetworkElementsDict
from pyzefir.model.network_elements import DemandChunk, EnergySource
from pyzefir.optimization.linopy.preprocessing.indices import IndexingSet, Indices
from pyzefir.optimization.linopy.preprocessing.variables.utils import add_h_y_variable


def create_dch_vars(
    indices: Indices,
    demand_chunks: NetworkElementsDict[DemandChunk],
    energy_sources: NetworkElementsDict[EnergySource],
    model: Model,
    energy_source_ii: IndexingSet,
    var_name: str,
) -> dict[int, dict[int, Variable]]:
    """
    Creates a dictionary of demand chunk variables.

    This function generates a nested dictionary mapping each demand chunk index to
    the associated energy source indices, creating variables for each combination.
    The variables represent the relationship between demand chunks and energy sources.

    Args:
        - indices (Indices): The indices object containing inverse mappings for demand chunks and energy sources.
        - demand_chunks (NetworkElementsDict[DemandChunk]): A dictionary containing the demand chunk elements.
        - energy_sources (NetworkElementsDict[EnergySource]): A dictionary containing the energy source elements.
        - model (Model): The optimization model to which the variables will be added.
        - energy_source_ii (IndexingSet): An indexing set for energy sources.
        - var_name (str): Base name for the variable, which will be used to create unique variable names.

    Returns:
        - dict[int, dict[int, Variable]]: A dictionary mapping each demand chunk index to another dictionary,
          which maps energy source indices to the created variables for each hour and year.
    """
    result: dict[int, dict[int, Variable]] = dict()
    for demand_chunk in demand_chunks.values():
        dch_idx = indices.DEMCH.inverse[demand_chunk.name]
        result[dch_idx] = dict()
        for energy_source in energy_sources.values():
            if demand_chunk.tag in energy_source.tags:
                energy_source_idx = energy_source_ii.inverse[energy_source.name]
                result[dch_idx][energy_source_idx] = add_h_y_variable(
                    model,
                    indices,
                    var_name=f"{var_name}_{demand_chunk.name}_{energy_source.name}",
                )
    return result
