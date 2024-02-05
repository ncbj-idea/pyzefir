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
from pyzefir.model.network_elements import EnergySourceType
from pyzefir.model.utils import check_interval

if TYPE_CHECKING:
    from pyzefir.model.network import Network


@dataclass(kw_only=True)
class StorageType(EnergySourceType):
    """
    A class that represents the StorageType in the network structure which stores parameters
    defined for a given type of storages
    """

    energy_type: str
    """
    Name of energy type stored by the StorageType
    """
    generation_efficiency: float
    """
    Parameter describes the fraction of stored energy losses during its use
    """
    load_efficiency: float
    """
    Parameter describes the fraction of stored energy losses during charging
    """
    cycle_length: int
    """
    Number of hours after which state of charge must be 0;
    if cycle_len = 10, then soc = 0 for hours: 0, 10, 20, ...
    """
    power_to_capacity: float
    """
    Ratio of storage power to storage capacity
    """
    energy_loss: float = 0.0
    """
    energy losses associated with soc
    """
    power_utilization: float
    """
    Determines the percentage of the installed generator's rated power that
    can be used
    """

    def validate(self, network: Network) -> None:
        """
        Validation procedure checking:
        - Validates types of the following attributes StorageType class:
        - generation_efficiency
        - load_efficiency
        - cycle_length
        - power_to_capacity

        Args:
            network (Network): network to which self is to be added.

        Raises:
            NetworkValidatorExceptionGroup: If exception_list contains exception.
        """
        exception_list: list[NetworkValidatorException] = []

        self._validate_energy_source_type_base(network, exception_list)
        for attr, attr_type in [
            ("generation_efficiency", float | int),
            ("load_efficiency", float | int),
            ("cycle_length", int),
            ("power_to_capacity", float | int),
            ("energy_type", str),
            ("energy_loss", float | int),
            ("power_utilization", float | int),
        ]:
            self._validate_attribute_type(
                attr=attr,
                attr_type=attr_type,
                exception_list=exception_list,
                raise_error=True,
            )
        if self.energy_type not in network.energy_types:
            exception_list.append(
                NetworkValidatorException(
                    f"StorageType {self.name} has energy type {self.energy_type}"
                    " which is not compliant with the network"
                    f" energy types: {sorted(network.energy_types)}"
                )
            )
        for attr in (
            "generation_efficiency",
            "load_efficiency",
            "energy_loss",
            "power_utilization",
        ):
            if not np.isnan(attr_value := getattr(self, attr)) and not check_interval(
                lower_bound=0, upper_bound=1, value=getattr(self, attr)
            ):
                exception_list.append(
                    NetworkValidatorException(
                        f"The value of the {attr} is inconsistent with th expected bounds of "
                        f"the interval: 0 <= {attr_value} <= 1"
                    )
                )

        if exception_list:
            raise NetworkValidatorExceptionGroup(
                f"While adding StorageType {self.name} following errors occurred: ",
                exception_list,
            )
