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
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.parameters import ModelParameters


@dataclass
class LineParameters(ModelParameters):
    """
    Class representing the line parameters for energy transmission systems.

    This class holds various parameters associated with transmission lines, such as energy types,
    bus connections, transmission losses, and capacity limits. It is essential for modeling
    the performance and characteristics of lines in energy distribution networks.
    """

    def __init__(self, lines: NetworkElementsDict, indices: Indices) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - lines (NetworkElementsDict): The dictionary of lines in the network.
            - indices (Indices): The indices used to map line properties.
        """
        self.et = self.fetch_element_prop(lines, indices.LINE, "energy_type")
        """ line energy type """
        self.bus_from = self.get_index_from_prop(lines, indices.LINE, indices.BUS, "fr")
        """ line starting bus """
        self.bus_to = self.get_index_from_prop(lines, indices.LINE, indices.BUS, "to")
        """ line end bus """
        self.loss = self.fetch_element_prop(lines, indices.LINE, "transmission_loss")
        """ line transmission loss """
        self.cap = self.fetch_element_prop(lines, indices.LINE, "max_capacity")
        """ line capacity [for now it is given apriori and can not change] """
        self.tf = {
            key: indices.TF.inverse[val]
            for key, val in self.fetch_element_prop(
                lines, indices.LINE, "transmission_fee"
            ).items()
            if val
        }
        """ line transmission fee """
