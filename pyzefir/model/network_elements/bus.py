from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network_element import NetworkElement

if TYPE_CHECKING:
    from pyzefir.model.network import Network

_logger = logging.getLogger(__name__)


@dataclass
class Bus(NetworkElement):
    """
    A class that represents the Bus element in the network structure
    """

    energy_type: str
    """
    Name of the bus energy type
    """
    generators: set[str] = field(default_factory=set, init=False)
    """
    Set of generators attached to the bus
    """
    storages: set[str] = field(default_factory=set, init=False)
    """
    Set of storages attached to the bus
    """
    lines_in: set[str] = field(default_factory=set, init=False)
    """
    Set of lines oriented positive (inflow orientation)
    """
    lines_out: set[str] = field(default_factory=set, init=False)
    """
    Set of lines oriented negative (outflow orientation)
    """

    def _validate_energy_type(
        self,
        network: Network,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        if not isinstance(self.energy_type, str):
            exception_list.append(
                NetworkValidatorException(
                    f"Energy Type must be a string, but given {type(self.energy_type)} instead"
                )
            )
        elif self.energy_type not in network.energy_types:
            exception_list.append(
                NetworkValidatorException(
                    f"Bus {self.name} has energy type {self.energy_type}"
                    " which is not compliant with the network"
                    f" energy types: {sorted(network.energy_types)}"
                )
            )

    def validate(self, network: Network) -> None:
        """
        Validation procedure checking:
        - whether the Bus energy_type is in the energy_type of the Network

        Args:
            network (Network): Network object to which this object belongs

        Raises:
            NetworkValidatorExceptionGroup: If any of the validation fails

        """
        exception_list: list[NetworkValidatorException] = []
        self._validate_name_type(exception_list)
        self._validate_energy_type(network, exception_list)
        if exception_list:
            raise NetworkValidatorExceptionGroup(
                f"While adding Bus {self.name} following errors occurred: ",
                exception_list,
            )

    def attach_generator(self, generator_name: str) -> None:
        """
        Adds given generator name to this Bus generator set.

        Args:
            generator_name (str): Generator name to attach
        """
        if generator_name in self.generators:
            _logger.debug(
                f"Generator name: {generator_name} already in {self.name} generators"
            )
            return None
        self.generators.add(generator_name)
        _logger.debug(
            f"Generator name: {generator_name} added to {self.name} generators"
        )

    def attach_storage(self, storage_name: str) -> None:
        """
        Adds given storage name to this Bus storage set.

        Args:
            storage_name (str): Storage name to attach
        """

        if storage_name in self.storages:
            _logger.debug(
                f"Storage name: {storage_name} already in {self.name} storages"
            )
            return None
        self.storages.add(storage_name)
        _logger.debug(f"Storage name: {storage_name} added to {self.name} storages")

    def attach_from_line(self, line_name: str) -> None:
        """
        Adds given line name to this Bus line "from" set.

        Args:
            line_name (str): Line name to attach
        """

        if line_name in self.lines_out:
            _logger.debug(f"Line name: {line_name} already in {self.name} line_out")
            return None
        self.lines_out.add(line_name)
        _logger.debug(f"Line name: {line_name} added to {self.name} line_out")

    def attach_to_line(self, line_name: str) -> None:
        """
        Adds given line name to this Bus line "to" set.

        Args:
            line_name (str): Line name to attach
        """

        if line_name in self.lines_in:
            _logger.debug(f"Line name: {line_name} already in {self.name} line_in")
            return None
        self.lines_in.add(line_name)
        _logger.debug(f"Line name: {line_name} added to {self.name} line_in")
