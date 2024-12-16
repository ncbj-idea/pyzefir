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
class TransmissionFeeParameters(ModelParameters):
    """
    Class representing the transmission fee parameters.

    This class encapsulates the parameters related to transmission fees, including
    hourly profiles and scaling factors. These parameters are essential for calculating
    the cost associated with energy transmission across the network.
    """

    def __init__(
        self,
        transmission_fees: NetworkElementsDict,
        indices: Indices,
        scale: float = 1.0,
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - transmission_fees (NetworkElementsDict): The transmission fee elements in the network.
            - indices (Indices): The indices used for mapping various parameters.
            - scale (float, optional): A scaling factor for the transmission fees. Defaults to 1.0.
        """
        self.fee = self.fetch_element_prop(
            transmission_fees, indices.TF, "fee", sample=indices.H.ii
        )
        """ transmission fee hourly profile; transmission_fee -> (h -> fee[y]) """
        self.fee = self.scale(self.fee, scale)  # noqa
