# PyZefir
# Copyright (C) 2023-2024 Narodowe Centrum Badań Jądrowych
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

import pandas as pd

from pyzefir.model.network_elements import DemandChunk
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser
from pyzefir.utils.path_manager import DataSubCategories


class DemandChunkParser(AbstractElementParser):
    def __init__(
        self,
        demand_chunk_dict: dict[str, pd.DataFrame],
    ) -> None:
        self.demand_chunk_dict = demand_chunk_dict

    def create(self) -> tuple[DemandChunk, ...]:
        demand_chunks: list[DemandChunk] = list()
        for _, demand_chunk_config in self.demand_chunk_dict[
            DataSubCategories.DEMAND_CHUNKS
        ].iterrows():
            demand_chunk = DemandChunk(
                name=demand_chunk_config["name"],
                tag=demand_chunk_config["tag"],
                energy_type=demand_chunk_config["energy_type"],
                periods=self.demand_chunk_dict[demand_chunk_config["name"]][
                    ["period_start", "period_end"]
                ].to_numpy(),
                demand=self.demand_chunk_dict[demand_chunk_config["name"]]
                .drop(columns=["period_start", "period_end"])
                .to_numpy(),
            )
            demand_chunks.append(demand_chunk)

        return tuple(demand_chunks)
