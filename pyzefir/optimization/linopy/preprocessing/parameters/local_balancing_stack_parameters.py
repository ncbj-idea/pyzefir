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

from pyzefir.model.network import NetworkElementsDict
from pyzefir.model.network_elements import AggregatedConsumer, LocalBalancingStack
from pyzefir.optimization.linopy.preprocessing.indices import IndexingSet, Indices
from pyzefir.optimization.linopy.preprocessing.parameters import ModelParameters


@dataclass
class LBSParameters(ModelParameters):
    """
    Class representing the local balancing stack parameters.

    This class holds parameters related to local balancing stacks, including the buses
    connected to these stacks and the aggregated consumer indices they are linked to.
    It is crucial for managing local balancing in energy distribution networks.
    """

    def __init__(
        self,
        local_balancing_stacks: NetworkElementsDict,
        aggregated_consumers: NetworkElementsDict,
        indices: Indices,
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - local_balancing_stacks (NetworkElementsDict): The dictionary of local balancing stacks.
            - aggregated_consumers (NetworkElementsDict): The dictionary of aggregated consumers.
            - indices (Indices): The indices used to map local balancing stack properties.
        """
        self.bus_out = self.get_bus_out(
            local_balancing_stacks, indices.BUS, indices.LBS
        )
        """ bus, for which aggregated consumer demand will be included in balancing equation """
        self.aggr_idx = self.get_aggr_idx(
            aggregated_consumers, indices.AGGR, indices.LBS
        )
        """ aggregated consumer, to which given local balancing stack is / can be connected; lbs_id -> aggr_id """
        self.buses = self.get_buses(local_balancing_stacks, indices.BUS, indices.LBS)

    @staticmethod
    def get_bus_out(
        local_balancing_stacks: NetworkElementsDict[LocalBalancingStack],
        bus_idx: IndexingSet,
        lbs_idx: IndexingSet,
    ) -> dict[int, dict[str, int]]:
        """
        Returns outgoing buses connected to the local balancing stack.

        Args:
            - local_balancing_stacks (NetworkElementsDict[LocalBalancingStack]): local balancing stack
            - bus_idx (IndexingSet): index of bus
            - lbs_idx (IndexingSet): index of local balancing stack

        Returns:
            - dict[int, dict[str, int]]: buses connected to the lbs
        """
        return {
            ii: {
                et: bus_idx.inverse[bus]
                for et, bus in local_balancing_stacks[name].buses_out.items()
            }
            for ii, name in lbs_idx.mapping.items()
        }

    @staticmethod
    def get_aggr_idx(
        aggregated_consumers: NetworkElementsDict[AggregatedConsumer],
        aggr_idx: IndexingSet,
        lbs_idx: IndexingSet,
    ) -> dict[int, int]:
        """
        Returns connected aggregators to the local balancing stack.

        Args:
            - aggregated_consumers (NetworkElementsDict[AggregatedConsumer]): aggregated consumers
            - aggr_idx (IndexingSet): index of aggregator
            - lbs_idx (IndexingSet): index of local balancing stack

        Returns:
            - dict[int, int]: dict of connected aggregators to the lbs
        """
        return {
            lbs_idx.inverse[lbs_name]: aggr_id
            for aggr_id, aggr_name in aggr_idx.mapping.items()
            for lbs_name in aggregated_consumers[aggr_name].stack_base_fraction
        }

    @staticmethod
    def get_buses(
        local_balancing_stacks: NetworkElementsDict[LocalBalancingStack],
        bus_idx: IndexingSet,
        lbs_idx: IndexingSet,
    ) -> dict[int, dict[str, set[int]]]:
        """
        Returns buses connected to the local balancing stack.

        Args:
            - local_balancing_stacks (NetworkElementsDict[LocalBalancingStack]): local balancing stacks
            - bus_idx (IndexingSet): index of bus
            - lbs_idx (IndexingSet): index of local balancing stack

        Returns:
            - dict[int, dict[str, set[int]]]: dict of buses connected to the lbs
        """
        return {
            ii: {
                et: {bus_idx.inverse[bus] for bus in buses}
                for et, buses in local_balancing_stacks[name].buses.items()
            }
            for ii, name in lbs_idx.mapping.items()
        }
