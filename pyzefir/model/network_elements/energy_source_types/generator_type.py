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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network_elements import EnergySourceType

if TYPE_CHECKING:
    from pyzefir.model.network import Network


@dataclass(kw_only=True)
class GeneratorType(EnergySourceType):
    """
    A class that represents the GeneratorType in the network structure which stores parameters
    defined for a given type of generators
    """

    efficiency: dict[str, float]
    """
    Efficiency defined for every energy type that is produced by the generator
    """
    energy_types: set[str]
    """
    Names of energy types produced by the GeneratorType
    """
    emission_reduction: dict[str, float]
    """
    Reduction of emission for specific emission type applied by the generator.
    """
    conversion_rate: dict[str, pd.Series] = field(default_factory=dict)
    """
    Conversion rate describes amount of produced energy from 1 unit of given energy type
    """
    fuel: str | None = field(default=None)
    """
    Name of the fuel that can be used by the generator (for dispatchable generators)
    """
    capacity_factor: str | None = field(default=None)
    """
    Name of the generator capacity factor (for non-dispatchable generators)
    """
    ramp: float = np.nan
    """Percentage difference between generations in relation to the installed capacity subsequent hours"""

    def validate(self, network: Network) -> None:
        """
        Validation procedure checking:
        - Validates whether at least one of capacity factor or fuel is not None

        Method validate runs following validate methods:
        - _validate_fuels
        - _validate_capacity_factor
        - _validate_efficiency
        - _validate_emission_reduction
        - _validate_conversion_rate

        Args:
            network (Network): network to which self is to be added

        Returns:
            None

        Raises:
            NetworkValidatorExceptionGroup: If exception_list contains exception.
        """
        exception_list: list[NetworkValidatorException] = []
        self._validate_energy_source_type_base(network, exception_list)
        if self.capacity_factor is not None and self.fuel is not None:
            exception_list.append(
                NetworkValidatorException(
                    f"Generator type {self.name} can have either capacity "
                    f"factor or fuel at the same time"
                )
            )
        self._validate_fuels(exception_list, network)
        self._validate_capacity_factor(exception_list, network)
        self._validate_efficiency(exception_list, network)
        self._validate_emission_reduction(exception_list, network)
        self._validate_conversion_rate(exception_list, network)
        if not isinstance(self.ramp, float | int):
            exception_list.append(
                NetworkValidatorException("Ramp value must be float or empty.")
            )
        elif not np.isnan(self.ramp) and not 0 < self.ramp < 1:
            exception_list.append(
                NetworkValidatorException(
                    f"Ramp value for {self.name} must be "
                    f"greater than 0 and less than 1, but it is {self.ramp}"
                )
            )
        if exception_list:
            raise NetworkValidatorExceptionGroup(
                f"While adding GeneratorType {self.name} following "
                f"errors occurred: ",
                exception_list,
            )

    @property
    def inbound_energy_type(self) -> set[str]:
        """
        Gets set of energy types needed by the generator
        Returns:
            set[str]: Set of energy types.
        """
        return set(self.conversion_rate) if self.conversion_rate else set()

    def _validate_fuels(
        self, exception_list: list[NetworkValidatorException], network: Network
    ) -> None:
        """
        Validation procedure checking:
        - Validates fuel type
        - Fuel reference validation

        Args:
            exception_list (NetworkValidatorException) - list of raised exceptions.
            network (Network): network to which self is to be added.

        Returns:
            None
        """
        if self.fuel is not None and not isinstance(self.fuel, str):
            exception_list.append(
                NetworkValidatorException(
                    "None or str type for fuel expected but type:"
                    f" {type(self.fuel)} for generator type: {self.name} given"
                )
            )
        if self.fuel is not None and self.fuel not in network.fuels:
            exception_list.append(
                NetworkValidatorException(
                    f"Generator {self.name} fuel {self.fuel} has not been added to the network"
                )
            )

    def _validate_capacity_factor(
        self, exception_list: list[NetworkValidatorException], network: Network
    ) -> None:
        """
        Validation procedure checking:
        - Validates if capacity_factor not string type or None
        - Validates if capacity_factor is not None and not exists in network.capacity_factors

        Args:
            exception_list (NetworkValidatorException) - list of raised exceptions.
            network (Network): network to which self is to be added.

        Returns:
            None
        """
        if (
            not isinstance(self.capacity_factor, str)
            and self.capacity_factor is not None
        ):
            exception_list.append(
                NetworkValidatorException(
                    f"None or str type for capacity factor expected but type: {type(self.capacity_factor)} "
                    f"for generator type: {self.name} given"
                )
            )
        if (
            self.capacity_factor is not None
            and self.capacity_factor not in network.capacity_factors
        ):
            exception_list.append(
                NetworkValidatorException(
                    f"Generator type '{self.name}' capacity factor "
                    f"'{self.capacity_factor}' has not been added to the network"
                )
            )

    def _validate_conversion_rate(
        self,
        exception_list: list[NetworkValidatorException],
        network: Network,
    ) -> None:
        """
        Validation procedure checking:
        - Validates whether conversion rate is not None and conversion rate energy types exist
        in network.energy_types

        Args:
            exception_list (NetworkValidatorException) - list of raised exceptions.
            network (Network): network to which self is to be added.

        Returns:
            None
        """
        if self.conversion_rate is None:
            exception_list.append(
                NetworkValidatorException(
                    f"Conversion rate of generator type: {self.name} cannot be None."
                )
            )
            return
        if not set(self.conversion_rate.keys()).issubset(network.energy_types):
            exception_list.append(
                NetworkValidatorException(
                    f"Conversion rate energy types of {self.name} do not exist "
                    f"in network energy types: {sorted(network.energy_types)}"
                )
            )

    def _validate_efficiency(
        self, exception_list: list[NetworkValidatorException], network: Network
    ) -> None:
        """
        Validation procedure checking:
        - Validates if efficiency of generator type is not None
        - Validates whether efficiency energy types exist in network.energy_types

        Args:
            exception_list (NetworkValidatorException) - list of raised exceptions.
            network (Network): network to which self is to be added.

        Returns:
            None
        """
        if self.efficiency is None:
            exception_list.append(
                NetworkValidatorException(
                    f"Efficiency of generator type: {self.name} cannot be None."
                )
            )
            return
        if not set(self.efficiency.keys()).issubset(network.energy_types):
            exception_list.append(
                NetworkValidatorException(
                    f"Efficiency energy types of {self.name} do not exist "
                    f"in network energy types: {sorted(network.energy_types)}"
                )
            )

    def _validate_emission_reduction(
        self, exception_list: list[NetworkValidatorException], network: Network
    ) -> None:
        """
        Validation procedure checking:
        - Validates if emission reduction of generator type is not None
        - Validates whether emission reduction energy types exist in network.emission_types

        Args:
            exception_list (NetworkValidatorException) - list of raised exceptions.
            network (Network): network to which self is to be added.

        Returns:
            None
        """
        if self.emission_reduction is None:
            exception_list.append(
                NetworkValidatorException(
                    f"Emission reduction of generator type: {self.name} cannot be None."
                )
            )
            return
        if emission_diff := set(self.emission_reduction.keys()).difference(
            network.emission_types
        ):
            exception_list.append(
                NetworkValidatorException(
                    f"Emission reduction emission types {emission_diff} of "
                    f"{self.name} do not exist "
                    f"in network emission types: {sorted(network.emission_types)}"
                )
            )
