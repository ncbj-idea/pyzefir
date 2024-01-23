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
class FuelParameters(ModelParameters):
    """Fuel parameters (coal, gas, biomass, etc.)"""

    def __init__(
        self, fuels: NetworkElementsDict, indices: Indices, scale: float = 1.0
    ) -> None:
        self.u_emission = self.fetch_element_prop(fuels, indices.FUEL, "emission")
        """ base emission per unit; fuel -> emission_type """
        self.energy_per_unit = self.fetch_element_prop(
            fuels, indices.FUEL, "energy_per_unit"
        )
        """ base energy per unit; fuel -> energy_per_unit """
        self.unit_cost = self.fetch_element_prop(
            fuels, indices.FUEL, "cost", sample=indices.Y.ii
        )
        """ cost per unit; fuel -> (y -> cost[y]) """
        self.unit_cost = self.scale(self.unit_cost, scale)  # noqa
        self.availability = self.fetch_element_prop(
            fuels, indices.FUEL, "availability", sample=indices.Y.ii
        )
        """ total availability per year; fuel -> (y -> availability[y]) """
