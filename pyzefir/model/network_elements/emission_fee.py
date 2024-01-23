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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network_element import NetworkElement
from pyzefir.model.utils import validate_series

if TYPE_CHECKING:
    from pyzefir.model.network import Network


@dataclass(kw_only=True)
class EmissionFee(NetworkElement):
    """
    A class that represents the Emission Fee element in the network structure
    """

    emission_type: str
    """ Name of the emission type """
    price: pd.Series
    """ Amount of a given emission fee in particular years"""

    def validate(self, network: Network) -> None:
        """
        Validation procedure checking:
        - if emission_type is in the network
        - if the price is a correct pd.Series

        Args:
            network (Network): Network to which EmissionFee is to be added.

        Raises:
            NetworkValidatorExceptionGroup: If any of the validations fails.
        """
        exception_list: list[NetworkValidatorException] = []

        if self.emission_type not in network.emission_types:
            exception_list.append(
                NetworkValidatorException(
                    f"Emission type: {self.emission_type} does not exist in the network"
                )
            )

        validate_series(
            name=f"EmissionFee {self.name}",
            series=self.price,
            length=network.constants.n_years,
            exception_list=exception_list,
            index_type=np.integer,
            values_type=np.floating,
            allow_null=False,
        )

        if exception_list:
            raise NetworkValidatorExceptionGroup(
                f"While adding EmissionFee {self.name} following errors occurred: ",
                exception_list,
            )
