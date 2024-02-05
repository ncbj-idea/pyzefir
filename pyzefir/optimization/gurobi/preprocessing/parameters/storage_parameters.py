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
from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.parameters import ModelParameters


@dataclass
class StorageParameters(ModelParameters):
    def __init__(
        self,
        storages: NetworkElementsDict,
        storage_types: NetworkElementsDict,
        indices: Indices,
    ) -> None:
        self.base_cap = self.fetch_element_prop(storages, indices.STOR, "unit_base_cap")
        """ storage base capacity """
        self.et = self.fetch_energy_source_type_prop(
            storages, storage_types, indices.STOR, "energy_type"
        )
        """ storage energy type """
        self.gen_eff = self.fetch_energy_source_type_prop(
            storages, storage_types, indices.STOR, "generation_efficiency"
        )
        """ storage generation efficiency """
        self.load_eff = self.fetch_energy_source_type_prop(
            storages, storage_types, indices.STOR, "load_efficiency"
        )
        """ storage load efficiency """
        self.p2cap = self.fetch_energy_source_type_prop(
            storages, storage_types, indices.STOR, "power_to_capacity"
        )
        """ power to capacity ratio """
        self.bus = self.get_index_from_prop(storages, indices.STOR, indices.BUS, "bus")
        """ storage bus """
        self.cycle_len = self.fetch_energy_source_type_prop(
            storages, storage_types, indices.STOR, "cycle_length"
        )
        """ length of the loading cycle """
        self.unit_max_capacity = self.fetch_element_prop(
            storages, indices.STOR, "unit_max_capacity", sample=indices.Y.ii
        )
        """ storage max capacity in a given year """
        self.unit_min_capacity = self.fetch_element_prop(
            storages, indices.STOR, "unit_min_capacity", sample=indices.Y.ii
        )
        """ storage min capacity in a given year """
        self.unit_max_capacity_increase = self.fetch_element_prop(
            storages, indices.STOR, "unit_max_capacity_increase", sample=indices.Y.ii
        )
        """ storage capacity increase upper bound of capacity increase in a given year """
        self.unit_min_capacity_increase = self.fetch_element_prop(
            storages, indices.STOR, "unit_min_capacity_increase", sample=indices.Y.ii
        )
        """ storage capacity increase lower bound of capacity increase in a given year """
        self.min_device_nom_power = self.get_prop_from_elements_if_not_none(
            storages, indices.STOR, "min_device_nom_power"
        )
        """ storage minimal device nominal power """
        self.max_device_nom_power = self.get_prop_from_elements_if_not_none(
            storages, indices.STOR, "max_device_nom_power"
        )
        """ storage maximum device nominal power """
        self.tstor = {
            i: indices.TSTOR.inverse[storages[gen].energy_source_type]
            for i, gen in indices.STOR.mapping.items()
        }
        """ storage type """
        self.tags = self.get_set_prop_from_element(
            storages, "tags", indices.STOR, indices.TAGS
        )
        """ storage tags """
