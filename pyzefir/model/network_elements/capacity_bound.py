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

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network_element import NetworkElement

if TYPE_CHECKING:
    from pyzefir.model.network import Network

_logger = logging.getLogger(__name__)


class CapacityBoundSenseLabel(StrEnum):
    """
    A class that represents the CapacityBoundsSenseLabel element in the network structure
    """

    EQ = "EQ"
    LEQ = "LEQ"


class CapacityBoundsExceptionGroup(NetworkValidatorExceptionGroup):
    pass


@dataclass(kw_only=True)
class CapacityBound(NetworkElement):
    """
    A class that represents the Capacity Bound element in the network structure
    """

    left_technology: str
    """
    Name of the technology (Generator | Storage) whose capacity will be used on the left side of the inequality
    """
    right_technology: str
    """
    Name of the technology (Generator | Storage) whose capacity will be used on the right side of the inequality
    """
    sense: str
    """
    Sense of inequality (leq = less than or equal, eq = equal)
    """
    left_coefficient: float
    """
    The coefficient used on the left side of the inequality
    """

    def validate(self, network: Network) -> None:
        """
        Validates the CapacityBound element
            - validate fields types
            - validate if left technologies in network gen | stor
            - validate if right technologies in network gen | stor
            - validate if sense in CapacityBoundSenseLabel
            - validate if left_coefficients in range 0.0 - 1.0

        Args:
            network (Network): Network to which Line is to be added.

        Raises:
            NetworkValidatorExceptionGroup: If any of the validations fails.
        """
        _logger.debug("Validating capacity bound element object: %s...", self.name)
        exception_list: list[NetworkValidatorException] = []
        self._validate_fields_types(exception_list)
        self._validate_technology(network, exception_list)
        self._validate_sense_value(exception_list)
        self._validate_left_coefficient(exception_list)

        if exception_list:
            _logger.debug("Got error validating capacity bound: %s", exception_list)
            raise CapacityBoundsExceptionGroup(
                f"While adding capacity bound {self.name} following errors occurred: ",
                exception_list,
            )
        _logger.debug("DSR %s validation: capacity bound", self.name)

    def _validate_fields_types(
        self, exception_list: list[NetworkValidatorException]
    ) -> None:
        for attr, attr_type in [
            ("left_technology", str),
            ("right_technology", str),
            ("sense", str),
            ("left_coefficient", float),
        ]:
            self._validate_attribute_type(
                attr=attr,
                attr_type=attr_type,
                exception_list=exception_list,
                raise_error=True,
            )

    def _validate_technology(
        self, network: Network, exception_list: list[NetworkValidatorException]
    ) -> None:
        for tech in (self.left_technology, self.right_technology):
            if tech not in network.generators and tech not in network.storages:
                exception_list.append(
                    NetworkValidatorException(
                        f"Technology name '{tech}' is not present in network generators or storages."
                    )
                )

    def _validate_sense_value(
        self, exception_list: list[NetworkValidatorException]
    ) -> None:
        valid_senses = [member.name for member in CapacityBoundSenseLabel]
        if self.sense not in valid_senses:
            exception_list.append(
                NetworkValidatorException(
                    f"The provided sense '{self.sense}' is not valid. Valid senses are: {valid_senses}."
                )
            )

    def _validate_left_coefficient(
        self, exception_list: list[NetworkValidatorException]
    ) -> None:
        if self.left_coefficient < 0.0 or self.left_coefficient > 1.0:
            exception_list.append(
                NetworkValidatorException(
                    f"The provided left coefficient '{self.left_coefficient}' is not valid. "
                    "It must be between <0.0 and 1.0>."
                )
            )
