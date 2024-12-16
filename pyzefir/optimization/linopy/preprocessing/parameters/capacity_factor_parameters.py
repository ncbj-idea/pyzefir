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
class CapacityFactorParameters(ModelParameters):
    """
    Class representing profiled carrier parameters (sun, wind, etc.).

    This class is designed to manage capacity factor profiles for different energy carriers, such as solar and wind.
    It fetches and stores hourly capacity factor profiles, allowing for the analysis of energy generation potential
    based on varying conditions throughout the day and year.
    """

    def __init__(self, capacity_factors: NetworkElementsDict, indices: Indices) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - capacity_factors (NetworkElementsDict): Dictionary containing capacity factor elements.
            - indices (Indices): Indices for the capacity factors.
        """
        self.profile = self.fetch_element_prop(
            capacity_factors, indices.CF, "profile", sample=indices.H.ii
        )
        """ capacity factor hourly profile; capacity_factor -> (h -> cf_profile[y]) """
