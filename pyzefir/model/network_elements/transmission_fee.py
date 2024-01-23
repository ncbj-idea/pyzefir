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
class TransmissionFee(NetworkElement):
    """
    A class that represents the TransmissionFee element in the network structure
    """

    fee: pd.Series
    """
    Hourly fee for transmission of energy
    """

    def validate(self, network: Network) -> None:
        """
        Validates the TransmissionFee element
            - if the fee is a correct pd.Series

        Args:
            network (Network): Network to which Line is to be added.

        Raises:
            NetworkValidatorExceptionGroup: If any of the validations fails.
        """
        exception_list: list[NetworkValidatorException] = []

        validate_series(
            name=f"TransmissionFee {self.name}",
            series=self.fee,
            length=network.constants.n_hours,
            exception_list=exception_list,
            allow_null=False,
        )

        if exception_list:
            raise NetworkValidatorExceptionGroup(
                f"While adding TransmissionFee {self.name} following errors occurred: ",
                exception_list,
            )
