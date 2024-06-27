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

import logging
from typing import TYPE_CHECKING

import pandas as pd

from pyzefir.model.utils import validate_series

if TYPE_CHECKING:
    from pyzefir.model.network import Network

from dataclasses import dataclass

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network_element import NetworkElement

_logger = logging.getLogger(__name__)


class GenerationFractionValidatorExceptionGroup(NetworkValidatorExceptionGroup):
    pass


@dataclass
class GenerationFraction(NetworkElement):
    tag: str
    """Identifier tag for generator | storage."""

    sub_tag: str
    """Sub-identifier tag for generator | storage"""

    fraction_type: str
    """Type of generation fraction, indicating the shape of the fraction (yearly, hourly)"""

    energy_type: str
    """Names of energy types produced by the GeneratorType"""

    min_generation_fraction: pd.Series
    """Minimum generation fraction as a Pandas Series in range [0, 1] in network years shape"""

    max_generation_fraction: pd.Series
    """Maximum generation fraction as a Pandas Series in range [0, 1] in network years shape"""

    def validate(self, network: Network) -> None:
        """
        Validation procedure checking:
        - if network doesn't contain any generator type and generator type is None


        Method validate runs following validate methods:
        - _validate_base_energy_source

        Args:
            network (Network): Network to which GenerationFraction is to be added.

        Raises:
            NetworkValidatorExceptionGroup: If GenerationFraction is invalid.
        """

        _logger.debug("Validating generation fraction object: %s...", self.name)
        exception_list: list[NetworkValidatorException] = []

        self._validate_types(exception_list)
        self._validate_energy_type(network, exception_list)
        self._validate_fraction_series(network, exception_list)
        self._validate_tags(network, exception_list)

        if exception_list:
            _logger.debug(
                "Got error validating generation fraction: %s", exception_list
            )
            raise GenerationFractionValidatorExceptionGroup(
                f"While adding generation fraction {self.name} following errors occurred: ",
                exception_list,
            )
        _logger.debug("Generation fraction %s validation: Done", self.name)

    def _validate_types(self, exception_list: list[NetworkValidatorException]) -> None:
        for attr, attr_type in [
            ("tag", str),
            ("sub_tag", str),
            ("fraction_type", str),
            ("energy_type", str),
            ("min_generation_fraction", pd.Series),
            ("max_generation_fraction", pd.Series),
        ]:
            self._validate_attribute_type(
                attr=attr,
                attr_type=attr_type,
                exception_list=exception_list,
                raise_error=True,
            )

    def _validate_fraction_series(
        self, network: Network, exception_list: list[NetworkValidatorException]
    ) -> None:
        for name, fraction_series in (
            ("min_generation_fraction", self.min_generation_fraction),
            ("max_generation_fraction", self.max_generation_fraction),
        ):
            validate_series(
                name=name,
                series=fraction_series,
                length=network.constants.n_years,
                exception_list=exception_list,
                values_type=float,
                allow_null=True,
            )
            if not (
                (fraction_series >= 0.0) & (fraction_series <= 1.0)
                | fraction_series.isna()
            ).all():
                exception_list.append(
                    NetworkValidatorException(
                        f"Invalid fraction series '{name}': all passed values must be between 0.0 and 1.0"
                    )
                )

    def _validate_energy_type(
        self, network: Network, exception_list: list[NetworkValidatorException]
    ) -> None:
        if self.energy_type not in network.energy_types:
            exception_list.append(
                NetworkValidatorException(
                    f"Energy type {self.energy_type} not found in the "
                    f"Network energy types: {[e for e in network.energy_types]}"
                )
            )

    def _validate_tags(
        self,
        network: Network,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        tags: set[str] = set()
        tags_energy_types: set[str] = set()
        for generator in network.generators.values():
            for tag in generator.tags:
                tags.add(tag)
            tags_energy_types.update(
                network.generator_types[generator.energy_source_type].energy_types
            )
        for storage in network.storages.values():
            for tag in storage.tags:
                tags.add(tag)
            tags_energy_types.add(
                network.storage_types[storage.energy_source_type].energy_type
            )

        for tag in (self.tag, self.sub_tag):
            if tag not in tags:
                exception_list.append(
                    NetworkValidatorException(
                        f"Provided tag {tag} not found in any generator or storage in Network."
                    )
                )
        if self.energy_type not in tags_energy_types:
            exception_list.append(
                NetworkValidatorException(
                    f"Energy type {self.energy_type} do not match to generator or storage in "
                    f"Network related with {self.tag} or {self.sub_tag}"
                )
            )
