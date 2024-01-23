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

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network_element import NetworkElement

if TYPE_CHECKING:
    from pyzefir.model.network import Network


@dataclass
class LocalBalancingStack(NetworkElement):
    """
    A class that represents the LocalBalancingStack element in the network structure.
    It consists of a set of buses, that are locally balancing together.
    There must be at least one bus for every energy type defined in the network.
    """

    buses_out: dict[str, str] = field(default_factory=dict)
    """
    Dictionary mapping energy type to bus to which Aggregated load is attached.
    For every energy type there must be at least one bus in LocalBalancingStack,
    which servers as outlet.
    """
    buses: dict[str, set[str]] = field(default_factory=dict)
    """
    Dictionary mapping energy type to set of all buses of given energy type contained in self.
    Not every bus have to be in buses_out.
    """

    def _validate_buses_out(
        self, network: Network, exception_list: list[NetworkValidatorException]
    ) -> None:
        """
        Validation procedure checking:
        - if buses_out is a correct type
        - if all buses_out are in the network
        - if all buses_out have correct energy type

        Args:
            network (Network): Network to which self is to be added
            exception_list (list[NetworkValidatorException]): List of exceptions to which new exceptions are added

        """
        if not isinstance(self.buses_out, dict):
            exception_list.append(
                NetworkValidatorException(
                    f"Outlet buses of local balancing stack {self.name} must be a dict, "
                    f"not {type(self.buses_out)}."
                )
            )
            return

        for energy_type, bus_name in self.buses_out.items():
            if not isinstance(bus_name, str):
                exception_list.append(
                    NetworkValidatorException(
                        f"Outlet bus name for energy type {energy_type} of local balancing stack {self.name} "
                        f"must be a string, not {type(bus_name)}."
                    )
                )
                return
            if not isinstance(energy_type, str):
                exception_list.append(
                    NetworkValidatorException(
                        f"Energy type for outlet bus {bus_name} of local balancing stack {self.name} "
                        f"must be a string, not {type(energy_type)}."
                    )
                )
                return

            if bus_name not in network.buses:
                exception_list.append(
                    NetworkValidatorException(
                        f"Bus {bus_name} which is declared as an outlet bus of "
                        f"local balancing stack {self.name} does not exist "
                        "in the network."
                    )
                )
            elif network.buses[bus_name].energy_type != energy_type:
                exception_list.append(
                    NetworkValidatorException(
                        f"Bus {bus_name} can not be declared as an outlet bus of "
                        f"local balancing stack {self.name} for energy "
                        f"{energy_type}, since its energy type is "
                        f"{network.buses[bus_name].energy_type}."
                    )
                )

    def _validate_buses_type(
        self, exception_list: list[NetworkValidatorException]
    ) -> None:
        """
        Validation procedure checking:
            - validates if self.buses is a dict
            - validates energy types' type
            - validates if buses names collection is set
            - validates buses names type

        Args:
            exception_list (list[NetworkValidatorException]): List of exceptions to which new exceptions are added

        """
        self._validate_attribute_type(
            attr="buses",
            attr_type=dict,
            exception_list=exception_list,
            raise_error=True,
        )
        if not all(isinstance(energy_types, str) for energy_types in self.buses.keys()):
            exception_list.append(
                NetworkValidatorException(
                    f"All the energy types (keys of buses dict) must be a string, "
                    f"but following types are found: {[type(x).__name__ for x in self.buses.keys()]}"
                )
            )

        if not all(
            isinstance(buses_name_set, set) for buses_name_set in self.buses.values()
        ):
            exception_list.append(
                NetworkValidatorException(
                    f"Buses names collection type (values of buses dict) "
                    f"must be a set of strings, but following types are found: "
                    f"{[type(x).__name__ for x in self.buses.values()]}"
                )
            )

        if not all(
            all(isinstance(bus_names, str) for bus_names in bus_names_collection)
            for bus_names_collection in self.buses.values()
        ):
            exception_list.append(
                NetworkValidatorException(
                    "Buses names collection type (values of buses dict) must contain strings only"
                )
            )

    def _validate_buses(
        self, network: Network, exception_list: list[NetworkValidatorException]
    ) -> None:
        """
        Validation procedure checking:
            - validates if buses energy types exist in the network
            - validates if buses exist in the network.buses
            - validates if bus' energy type match with the bus' energy type in network

        Args:
            network (Network): Network to which self is to be added
            exception_list (list[NetworkValidatorException]): List of exceptions to which new exceptions are added

        Raises:
            NetworkValidatorException: If buses attribute has not been set for LocalBalancingStack object
        """
        self._validate_buses_type(exception_list)
        for energy_type in self.buses.keys():
            if energy_type not in network.energy_types:
                exception_list.append(
                    NetworkValidatorException(
                        f"Buses energy type {energy_type} is not defined in the Network"
                    )
                )
        for energy_type, buses in self.buses.items():
            for bus_name in buses:
                if bus_name not in network.buses:
                    exception_list.append(
                        NetworkValidatorException(
                            f"Bus name '{bus_name}' must exist in the Network"
                        )
                    )
                elif not energy_type == network.buses[bus_name].energy_type:
                    exception_list.append(
                        NetworkValidatorException(
                            f"Energy type for {bus_name} in {self.name} must match "
                            f"with energy type for the same bus in Network"
                        )
                    )

    def validate(self, network: Network) -> None:
        """
        Validate LocalBalancingStack element.
            - Validate buses out mapping

        Args:
            network (Network): Network to which LocalBalancingStack is
                to be added.

        Raises:
            NetworkValidatorExceptionGroup: If LocalBalancingStack is invalid.
        """
        exception_list: list[NetworkValidatorException] = []
        self._validate_name_type(exception_list)
        self._validate_buses_out(network=network, exception_list=exception_list)
        self._validate_buses(network=network, exception_list=exception_list)
        if exception_list:
            raise NetworkValidatorExceptionGroup(
                f"While adding Local Balancing Stack {self.name} following errors occurred: ",
                exception_list,
            )
