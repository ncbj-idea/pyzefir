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
class EmissionFeeParameters(ModelParameters):
    """
    Class representing the emission fee parameters.

    This class manages the parameters associated with emission fees, including the types of emissions
    and their corresponding prices. It allows for scaling of prices to accommodate different scenarios.
    """

    def __init__(
        self,
        emission_fees: NetworkElementsDict,
        indices: Indices,
        scale: float = 1.0,
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - emission_fees (NetworkElementsDict): Dictionary containing emission fee elements.
            - indices (Indices): Indices for accessing emission fee parameters.
            - scale (float, optional): Scaling factor for emission prices. Defaults to 1.0.
        """
        self.emission_type: dict[int, str] = self.fetch_element_prop(
            d=emission_fees, II=indices.EMF, prop="emission_type"
        )
        self.price: dict[int, np.ndarray] = self.fetch_element_prop(
            d=emission_fees, II=indices.EMF, prop="price", sample=indices.Y.ii
        )
        self.price = self.scale(self.price, scale)
