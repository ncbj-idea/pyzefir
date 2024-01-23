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

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network_element import NetworkElement

if TYPE_CHECKING:
    from pyzefir.model.network import Network


@dataclass
class DemandChunk(NetworkElement):
    """
    Demand chunk parameters
    """

    tag: str
    """
    Demand chunk tag
    """
    energy_type: str
    """
    Demand chunk energy type
    """
    periods: np.ndarray
    """
    Periods array with shape (n_periods, 2) where each row contains hour index of period_start and period_end
    """
    demand: np.ndarray
    """
    2D matrix with shape (n_periods, n_years), indicates energy demand for every period in every year
    """

    def _validate_periods_and_demand(
        self,
        exception_list: list[NetworkValidatorException],
        network: Network,
    ) -> None:
        """
        Validate periods and demand parameters
            - check if periods and demand have the same length
            - check if periods have 2 columns
            - check if periods are in the format (start, end)
            - check if demand has n_years columns
            - check if periods are type of int
            - check if demand is type of float

        Args:
            exception_list (list[NetworkValidatorException]): List of exceptions
                to which new exceptions will be added
            network (Network): Network object to which this object belongs
        """
        if len(self.periods) != len(self.demand):
            exception_list.append(
                NetworkValidatorException(
                    f"Length of periods ({len(self.periods)}) and demand ({len(self.demand)}) should be the same"
                )
            )

        if self.periods.shape[1] != 2:
            exception_list.append(
                NetworkValidatorException(
                    f"Periods should have 2 columns, not {self.periods.shape[1]}"
                )
            )
        elif not np.array(self.periods[:, 0] < self.periods[:, 1]).all():
            exception_list.append(
                NetworkValidatorException(
                    "Periods should be in the format (start, end), not (end, start)"
                )
            )

        if self.demand.shape[1] != network.constants.n_years:
            exception_list.append(
                NetworkValidatorException(
                    f"Demand should have {network.constants.n_years} columns, not {self.demand.shape[1]}"
                )
            )

        if not np.issubdtype(self.periods.dtype, np.integer):
            exception_list.append(
                NetworkValidatorException(
                    f"Periods should be type of int, not {self.periods.dtype}"
                )
            )

        if not np.issubdtype(self.demand.dtype, np.floating) and not np.issubdtype(
            self.demand.dtype, np.integer
        ):
            exception_list.append(
                NetworkValidatorException(
                    f"Demand should be type of float, not {self.demand.dtype}"
                )
            )

    def validate(self, network: Network) -> None:
        """
        Validate DemandChunk.
            - validates if tag is correct type and present in network
            - validates if energy type is correct type and present in network
            - validates periods attribute
            - validates demand attribute

        Args:
            network (Network): Network object to which this object belongs

        Raises:
            NetworkValidatorExceptionGroup: If any of the validation fails
        """
        exception_list: list[NetworkValidatorException] = []

        if isinstance(self.tag, str):
            if self.tag not in {
                *[tag for gen in network.generators.values() for tag in gen.tags],
                *[tag for stor in network.storages.values() for tag in stor.tags],
            }:
                exception_list.append(
                    NetworkValidatorException(
                        f"Tag {self.tag} is not defined in the network"
                    )
                )
        else:
            exception_list.append(
                NetworkValidatorException(
                    f"Tag {self.tag} should be type of str, not {type(self.tag).__name__}"
                )
            )

        if isinstance(self.energy_type, str):
            if self.energy_type not in network.energy_types:
                exception_list.append(
                    NetworkValidatorException(
                        f"Energy type {self.energy_type} is not defined in the network"
                    )
                )
        else:
            exception_list.append(
                NetworkValidatorException(
                    f"Energy type {self.energy_type} should be type of str, not {type(self.energy_type).__name__}"
                )
            )

        self._validate_periods_and_demand(exception_list, network)

        if exception_list:
            raise NetworkValidatorExceptionGroup(
                f"While adding DemandChunk {self.name} " "following errors occurred: ",
                exception_list,
            )
