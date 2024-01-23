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


@dataclass
class CapacityFactor(NetworkElement):
    """
    A class that represents the CapacityFactor in the network structureCapacity which defines generation profile
    from 1 unit of power for various non-dispatchable generators (i.e. pv, wind, ...).
    """

    profile: pd.Series
    """
    An hourly data series representing capacity factor
    """

    def validate(self, network: Network) -> None:
        """
        Validation procedure checking:
        - if profile is not none and is correct type

        Args:
            network (Network): network to which the CapacityFactor belongs
        """
        exception_list: list[NetworkValidatorException] = []

        self._validate_name_type(exception_list)
        validate_series(
            name="Profile",
            series=self.profile,
            length=network.constants.n_hours,
            exception_list=exception_list,
            allow_null=False,
        )

        if exception_list:
            raise NetworkValidatorExceptionGroup(
                f"While adding Capacity Factor {self.name} following errors occurred: ",
                exception_list,
            )
