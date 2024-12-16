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

from dataclasses import dataclass

import numpy as np

from pyzefir.model.network import NetworkElementsDict
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.parameters import ModelParameters


@dataclass
class DemandChunkParameters(ModelParameters):
    """
    Class representing the demand chunk parameters.

    This class encapsulates the parameters associated with demand chunks, which define specific energy demands
    over designated time periods. It retrieves and organizes information such as energy types, tags,
    demand amounts, and their respective periods for further analysis and modeling.
    """

    def __init__(
        self,
        demand_chunks: NetworkElementsDict,
        indices: Indices,
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - demand_chunks (NetworkElementsDict): Dictionary containing demand chunk elements.
            - indices (Indices): Indices for the demand chunks.
        """
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
        """
        Returns a dict of demand chunk ids and their corresponding demand for given periods.

        This method filters the specified periods to only include those that fall within the provided hour samples.
        It helps in identifying the applicable demand periods based on the hour data.

        Args:
            - periods (dict[int, np.ndarray]): periods of interest
            - hour_sample (nd.array): hour sample from input data

        Returns:
            - dict[int, np.ndarray]: sample periods of demands
        """
        result = dict()
        for demand_chunk_idx, period_data in periods.items():
            in_hour_sample = [
                interval
                for interval in period_data
                if set(range(*interval)).issubset(set(hour_sample))
            ]
            result[demand_chunk_idx] = np.array(in_hour_sample)
        return result
