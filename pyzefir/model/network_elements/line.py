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

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network_element import NetworkElement
from pyzefir.model.utils import check_interval

if TYPE_CHECKING:
    from pyzefir.model.network import Network


@dataclass(kw_only=True)
class Line(NetworkElement):
    """
    A class that represents the Line element in the network structure
    """

    energy_type: str
    """
    Unique name of Line's energy type
    """
    fr: str
    """
    Unique name of Bus from which this line departs
    """
    to: str
    """
    Unique name of Bus this line enters
    """
    transmission_loss: float
    """
    Losses of energy on a line parameter from [0,1]
    """
    max_capacity: float
    """
    Maximal amount of energy, that can flow through the line in one hour
    """
    transmission_fee: str | None = None
    """
    Name of the TransmissionFee element that defines the fee for transmission of energy
    """

    def _validate_energy_type(
        self,
        network: Network,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        if self.energy_type not in network.energy_types:
            exception_list.append(
                NetworkValidatorException(
                    f"Energy type of line {self.energy_type} not found in the "
                    f"Network energy types: {sorted([e for e in network.energy_types])}"
                )
            )

    def _validate_line_connections(
        self,
        network: Network,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        def _validate_line_connection(
            connected_bus_name: str,
            line_type_name: str,
        ) -> bool:
            """

            Returns: True if line bus is in Network buses

            """
            if not isinstance(connected_bus_name, str):
                exception_list.append(
                    NetworkValidatorException(
                        f"Line {line_type_name.capitalize()} must be of type string"
                    )
                )
                return False
            if connected_bus_name not in network.buses:
                exception_list.append(
                    NetworkValidatorException(
                        f"Cannot set the {line_type_name} of the line "
                        f"{self.name} to bus "
                        f"{connected_bus_name}. Bus {connected_bus_name} "
                        f"does not exist in the network"
                    )
                )
                return False
            if network.buses[connected_bus_name].energy_type != self.energy_type:
                exception_list.append(
                    NetworkValidatorException(
                        f"Cannot set {line_type_name} of the line {self.name} "
                        f"to bus {connected_bus_name}. "
                        f"Bus {connected_bus_name} energy type is "
                        f"{network.buses[connected_bus_name].energy_type}, "
                        f"which is different from the line {self.name} "
                        f"energy type: {self.energy_type}."
                    )
                )

            return True

        is_line_fr_connected = _validate_line_connection(self.fr, "beginning")
        is_line_to_connected = _validate_line_connection(self.to, "end")

        if (
            is_line_fr_connected
            and is_line_to_connected
            and network.buses[self.fr].energy_type != network.buses[self.to].energy_type
        ):
            exception_list.append(
                NetworkValidatorException(
                    f"Cannot add a line {self.name} between buses {self.fr} and "
                    f"{self.to} with different energy types "
                    f"{network.buses[self.fr].energy_type} != "
                    f"{network.buses[self.to].energy_type}"
                )
            )

    def _validate_transmission_loss(
        self,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        if not isinstance(self.transmission_loss, float | int):
            exception_list.append(
                NetworkValidatorException(
                    f"Transmission loss must be of type float, but is {type(self.transmission_loss)} instead"
                )
            )
            return None

        if not check_interval(
            lower_bound=0, upper_bound=1, value=self.transmission_loss
        ):
            exception_list.append(
                NetworkValidatorException(
                    f"The value of the transmission_loss is inconsistent with th expected bounds of "
                    f"the interval: 0 <= {self.transmission_loss} <= 1"
                )
            )

    def _validate_max_capacity(
        self,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        if not isinstance(self.max_capacity, float | int):
            exception_list.append(
                NetworkValidatorException(
                    f"Max capacity must be of type float, but is {type(self.max_capacity)} instead"
                )
            )

    def validate(self, network: Network) -> None:
        """
        Validate Line element.
            - if attributes have correct types
            - if energy type is in network's energy types
            - if line has proper fr and to connections:
                - connected bus is in the network
                - bus energy type is compliant with line energy type
            - if transmission loss has value between 0 and 1
            - if transmission fee is in network's transmission fees

        Args:
            network (Network): Network to which Line is to be added.

        Raises:
            NetworkValidatorExceptionGroup: If Line is invalid.
        """
        exception_list: list[NetworkValidatorException] = []

        self._validate_name_type(exception_list)
        self._validate_energy_type(network, exception_list)
        self._validate_line_connections(network, exception_list)
        self._validate_transmission_loss(exception_list)
        self._validate_max_capacity(exception_list)

        if (
            self.transmission_fee is not None
            and self.transmission_fee not in network.transmission_fees
        ):
            exception_list.append(
                NetworkValidatorException(
                    f"Cannot set a transmission fee for the line {self.name}. "
                    f"Transmission fee {self.transmission_fee} does not exist in the network"
                )
            )

        if exception_list:
            raise NetworkValidatorExceptionGroup(
                f"While adding Line {self.name} following errors occurred: ",
                exception_list,
            )
