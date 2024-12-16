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
from pyzefir.model.network_elements import Bus, LocalBalancingStack
from pyzefir.optimization.linopy.preprocessing.indices import IndexingSet, Indices
from pyzefir.optimization.linopy.preprocessing.parameters import ModelParameters


@dataclass
class BusParameters(ModelParameters):
    """
    Class representing parameters related to buses in an energy network.

    This class encapsulates properties and configurations of buses, including
    their energy types, associated generators, storage systems, and mappings
    to local balancing stacks.
    """

    def __init__(
        self,
        buses: NetworkElementsDict[Bus],
        local_balancing_stacks: NetworkElementsDict[LocalBalancingStack],
        indices: Indices,
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - buses (NetworkElementsDict[Bus]): A dictionary mapping bus names
                to their respective Bus objects.
            - local_balancing_stacks (NetworkElementsDict[LocalBalancingStack]):
                A dictionary mapping local balancing stack names to their respective LocalBalancingStack objects.
            - indices (Indices): An object containing indices for buses,
                generators, storage systems, lines, local balancing stacks, and demand-side responses.
        """
        self.et = self.fetch_element_prop(buses, indices.BUS, "energy_type")
        """ bus energy type """
        self.generators = self.get_set_prop_from_element(
            elements=buses,
            element_idx=indices.BUS,
            prop_idx=indices.GEN,
            prop="generators",
        )
        """ bus generators """
        self.storages = self.get_set_prop_from_element(
            elements=buses,
            element_idx=indices.BUS,
            prop_idx=indices.STOR,
            prop="storages",
        )
        """ bus storages """
        self.lines_in = self.get_set_prop_from_element(
            elements=buses,
            element_idx=indices.BUS,
            prop_idx=indices.LINE,
            prop="lines_in",
        )
        """ bus lines in """
        self.lines_out = self.get_set_prop_from_element(
            elements=buses,
            element_idx=indices.BUS,
            prop_idx=indices.LINE,
            prop="lines_out",
        )
        """ bus lines out """
        self.lbs_mapping = self.get_lbs_mapping(
            local_balancing_stacks, indices.BUS, indices.LBS
        )
        """ mapping bus_idx -> lbs_idx """
        self.dsr_type = self.get_index_from_prop_if_not_none(
            elements=buses,
            element_idx=indices.BUS,
            idx_to_get=indices.DSR,
            prop="dsr_type",
        )
        """ mapping bus_idx -> dsr_idx """

    @staticmethod
    def get_lbs_mapping(
        local_balancing_stacks: NetworkElementsDict[LocalBalancingStack],
        bus_idx: IndexingSet,
        lbs_idx: IndexingSet,
    ) -> dict[int, int]:
        """
        Returns mappings of bus indices to their corresponding local balancing stack indices.

        This method constructs a mapping that links each bus to its associated
        local balancing stacks based on the outgoing bus connections of each
        local balancing stack.

        Args:
            - local_balancing_stacks (NetworkElementsDict[LocalBalancingStack]):
              A dictionary mapping local balancing stack names to their respective
              LocalBalancingStack objects.
            - bus_idx (IndexingSet): Indexing set for buses, containing the mapping
              of bus indices.
            - lbs_idx (IndexingSet): Indexing set for local balancing stacks,
              containing the mapping of local balancing stack indices.

        Returns:
            - Dict[int, int]: A dictionary where keys are bus indices and
              values are the corresponding local balancing stack indices.
        """
        return {
            bus_idx.inverse[bus_name]: lbs_id
            for lbs_id, lbs_name in lbs_idx.mapping.items()
            for bus_name in local_balancing_stacks[lbs_name].buses_out.values()
        }
