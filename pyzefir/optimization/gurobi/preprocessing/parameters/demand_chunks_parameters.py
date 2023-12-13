from dataclasses import dataclass

import numpy as np

from pyzefir.model.network import NetworkElementsDict
from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.parameters import ModelParameters


@dataclass
class DemandChunkParameters(ModelParameters):
    """Generator parameters"""

    def __init__(
        self,
        demand_chunks: NetworkElementsDict,
        indices: Indices,
    ) -> None:
        self.energy_type = self.fetch_element_prop(
            demand_chunks, indices.DEMCH, "energy_type"
        )
        "energy type"
        self.tag = {
            k: indices.TAGS.inverse[v]
            for k, v in self.fetch_element_prop(
                demand_chunks, indices.DEMCH, "tag"
            ).items()
        }
        "tags involved"
        self.demand = self.fetch_element_prop(demand_chunks, indices.DEMCH, "demand")
        "demand to cover"
        self.periods = self.fetch_element_prop(demand_chunks, indices.DEMCH, "periods")
        self.periods = self.sample_periods(self.periods, indices.H.ii)
        """ time periods """

    @staticmethod
    def sample_periods(
        periods: dict[int, np.ndarray], hour_sample: np.ndarray
    ) -> dict[int, np.ndarray]:
        result = dict()
        for demand_chunk_idx, period_data in periods.items():
            in_hour_sample = [
                interval
                for interval in period_data
                if set(range(*interval)).issubset(set(hour_sample))
            ]
            result[demand_chunk_idx] = np.array(in_hour_sample)
        return result
