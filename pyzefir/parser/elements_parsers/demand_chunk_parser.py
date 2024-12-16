import pandas as pd

from pyzefir.model.network_elements import DemandChunk
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser
from pyzefir.utils.path_manager import DataSubCategories


class DemandChunkParser(AbstractElementParser):
    """
    Parses and processes data to create instances of DemandChunk.

    This class takes a dictionary of DataFrames containing demand chunk configurations
    and processes them to generate DemandChunk instances. Each demand chunk is constructed
    from the configuration data and corresponding demand data, ensuring all necessary attributes
    are set correctly.
    """

    def __init__(
        self,
        demand_chunk_dict: dict[str, pd.DataFrame],
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - demand_chunk_dict (dict[str, pd.DataFrame]): A dictionary mapping demand chunk names
              to their configuration DataFrames.
        """
        self.demand_chunk_dict = demand_chunk_dict

    def create(self) -> tuple[DemandChunk, ...]:
        """
        Creates and returns a tuple of DemandChunk instances.

        This method processes the demand chunk configuration DataFrame to create DemandChunk
        objects. It extracts the necessary attributes such as name, tag, energy type, periods,
        and demand data for each demand chunk, assembling them into a tuple for return.

        Returns:
            - tuple[DemandChunk, ...]: A tuple containing the created DemandChunk instances.
        """
        demand_chunks: list[DemandChunk] = list()
        for _, demand_chunk_config in self.demand_chunk_dict[
            DataSubCategories.DEMAND_CHUNKS
        ].iterrows():
            demand_chunk = DemandChunk(
                name=str(demand_chunk_config["name"]),
                tag=str(demand_chunk_config["tag"]),
                energy_type=str(demand_chunk_config["energy_type"]),
                periods=self.demand_chunk_dict[demand_chunk_config["name"]][
                    ["period_start", "period_end"]
                ].to_numpy(),
                demand=self.demand_chunk_dict[demand_chunk_config["name"]]
                .drop(columns=["period_start", "period_end"])
                .to_numpy(),
            )
            demand_chunks.append(demand_chunk)

        return tuple(demand_chunks)
