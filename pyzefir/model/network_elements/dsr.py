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

from __future__ import annotations

from typing import TYPE_CHECKING

from pyzefir.model.utils import check_interval

if TYPE_CHECKING:
    from pyzefir.model.network import Network

from dataclasses import dataclass, fields

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network_element import NetworkElement


@dataclass
class DSR(NetworkElement):
    name: str
    """
    Names of individual DSR
    """
    compensation_factor: float
    """
    Compensation factor in the both closed range [0,1]
    """
    balancing_period_len: int
    """
    Number of hours determining the length of the balancing period of DSR
    """
    penalization: float
    """
    Penalization of demand shift per unit of energy [PLN / xWh]
    """
    relative_shift_limit: float | None = None
    """
    Maximum total shift against total net injection at a node during this period,
    in the both opened range (0,1)
    """
    abs_shift_limit: float | None = None
    """
    Maximum total shift per period
    """

    def validate(self, network: Network) -> None:
        """
        Validation procedure checking:
        - All attributes have correct types
        - Compensation factor value in the both closed range [0,1]
        - Relative shift limit value in the both opened range (0,1)

        Args:
            network (Network): Network object to which this object belongs

        Raises:
            NetworkValidatorExceptionGroup: If any of the validation fails

        """
        exception_list: list[NetworkValidatorException] = []

        self._validate(exception_list)

        if exception_list:
            raise NetworkValidatorExceptionGroup(
                f"While adding DSR {self.name} following errors occurred: ",
                exception_list,
            )

    def _validate(self, exception_list: list[NetworkValidatorException]) -> None:
        for field in fields(self):
            self._validate_attribute_type(
                field.name, eval(str(field.type)), exception_list
            )

        if self.compensation_factor is not None and not check_interval(
            lower_bound=0, upper_bound=1, value=self.compensation_factor
        ):
            exception_list.append(
                NetworkValidatorException(
                    f"The value of the compensation_factor is inconsistent with th expected bounds of "
                    f"the interval: 0 <= {self.compensation_factor} <= 1"
                )
            )

        if self.relative_shift_limit is not None and not check_interval(
            lower_bound=0, upper_bound=1, value=self.relative_shift_limit
        ):
            exception_list.append(
                NetworkValidatorException(
                    f"The value of the relative_shift_limit is inconsistent with th expected bounds of "
                    f"the interval: 0 < {self.relative_shift_limit} < 1"
                )
            )
