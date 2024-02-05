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
class StorageTypeParameters(ModelParameters):
    """Storage Type parameters"""

    def __init__(
        self,
        storage_types: NetworkElementsDict,
        indices: Indices,
        scale: float = 1.0,
    ) -> None:
        self.max_capacity = self.fetch_element_prop(
            storage_types, indices.TSTOR, "max_capacity", sample=indices.Y.ii
        )
        """ storage max capacity in a given year """
        self.min_capacity = self.fetch_element_prop(
            storage_types, indices.TSTOR, "min_capacity", sample=indices.Y.ii
        )
        """ storage min capacity in a given year """
        self.max_capacity_increase = self.fetch_element_prop(
            storage_types, indices.TSTOR, "max_capacity_increase", sample=indices.Y.ii
        )
        """ storage capacity increase upper bound of capacity increase in a given year """
        self.min_capacity_increase = self.fetch_element_prop(
            storage_types, indices.TSTOR, "min_capacity_increase", sample=indices.Y.ii
        )
        """ storage capacity increase lower bound of capacity increase in a given year """
        self.capex = self.scale(
            self.fetch_element_prop(  # noqa
                storage_types, indices.TSTOR, "capex", sample=indices.Y.ii
            ),
            scale,
        )
        """ storage type capex per capacity unit """
        self.opex = self.scale(
            self.fetch_element_prop(
                storage_types, indices.TSTOR, "opex", sample=indices.Y.ii
            ),
            scale,
        )
        """ storage type opex per capacity unit """
        self.lt = self.fetch_element_prop(
            storage_types,
            indices.TSTOR,
            "life_time",
        )
        """ storage type life time """
        self.bt = self.fetch_element_prop(
            storage_types,
            indices.TSTOR,
            "build_time",
        )
        """ storage type build time """
        self.energy_loss = self.fetch_element_prop(
            storage_types, indices.TSTOR, "energy_loss", sample=indices.Y.ii
        )
        self.tags = self.get_set_prop_from_element(
            storage_types, "tags", indices.TSTOR, indices.T_TAGS
        )
        """ storage type tags """
        self.power_utilization = self.fetch_element_prop(
            storage_types, indices.TSTOR, "power_utilization"
        )
        """ storage power utilization """
