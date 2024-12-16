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

from pyzefir.model.network import NetworkElementsDict
from pyzefir.model.network_elements import DemandChunk, EnergySource
from pyzefir.optimization.linopy.preprocessing.indices import IndexingSet, Indices


def get_demand_chunk_unit_indices(
    indices: Indices,
    energy_sources: NetworkElementsDict[EnergySource],
    energy_source_ii: IndexingSet,
    demand_chunks: NetworkElementsDict[DemandChunk],
) -> dict[int, set[int]]:
    """
    Get demand chunk unit (generators or storages) indices.

    This function retrieves the indices of demand chunks associated with given energy sources.
    It maps energy source indices to sets of demand chunk indices based on tags.

    Args:
        - indices (Indices): Object containing inverse mappings for demand chunks.
        - energy_sources (NetworkElementsDict[EnergySource]): Dictionary of energy sources network elements.
        - energy_source_ii (IndexingSet): Object for energy sources.
        - demand_chunks (NetworkElementsDict[DemandChunk]): Dictionary of demand chunks network elements.

    Returns:
        - dict[int, set[int]]: Dictionary mapping energy source indices to sets of demand chunk indices.
    """

    result: dict[int, set[int]] = dict()
    for energy_source in energy_sources.values():
        energy_source_dem_ch_indices = {
            indices.DEMCH.inverse[dem_chunk.name]
            for dem_chunk in demand_chunks.values()
            if dem_chunk.tag in energy_source.tags
        }
        if energy_source_dem_ch_indices:
            result[energy_source_ii.inverse[energy_source.name]] = (
                energy_source_dem_ch_indices
            )
    return result
