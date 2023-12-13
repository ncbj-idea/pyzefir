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
