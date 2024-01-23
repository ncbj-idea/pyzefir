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

import numpy as np

from pyzefir.model.network import NetworkElementsDict
from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.parameters import ModelParameters


@dataclass
class EmissionFeeParameters(ModelParameters):
    """Emission fee parameters"""

    def __init__(
        self,
        emission_fees: NetworkElementsDict,
        indices: Indices,
        scale: float = 1.0,
    ) -> None:
        self.emission_type: dict[int, str] = self.fetch_element_prop(
            d=emission_fees, II=indices.EMF, prop="emission_type"
        )
        self.price: dict[int, np.ndarray] = self.fetch_element_prop(
            d=emission_fees, II=indices.EMF, prop="price", sample=indices.Y.ii
        )
        self.price = self.scale(self.price, scale)
