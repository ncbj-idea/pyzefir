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

from dataclasses import dataclass

from pyzefir.model.network import NetworkElementsDict
from pyzefir.model.network_elements import Bus, LocalBalancingStack
from pyzefir.optimization.gurobi.preprocessing.indices import IndexingSet, Indices
from pyzefir.optimization.gurobi.preprocessing.parameters import ModelParameters


@dataclass
class BusParameters(ModelParameters):
    def __init__(
        self,
        buses: NetworkElementsDict[Bus],
        local_balancing_stacks: NetworkElementsDict[LocalBalancingStack],
        indices: Indices,
    ) -> None:
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
        return {
            bus_idx.inverse[bus_name]: lbs_id
            for lbs_id, lbs_name in lbs_idx.mapping.items()
            for bus_name in local_balancing_stacks[lbs_name].buses_out.values()
        }
