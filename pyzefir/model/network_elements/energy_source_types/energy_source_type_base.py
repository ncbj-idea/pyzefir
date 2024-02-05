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

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from pyzefir.model.exceptions import NetworkValidatorException
from pyzefir.model.network_element import NetworkElement
from pyzefir.model.utils import validate_series

if TYPE_CHECKING:
    from pyzefir.model.network import Network


@dataclass(kw_only=True)
class EnergySourceType(NetworkElement, ABC):
    """
    An abstract class that defines the core attributes for each energy source type
    """

    name: str
    """
    Unique type name
    """
    life_time: int
    """
    Life time of added/installed capacity
    """
    build_time: int
    """
    Number of years needed to build new capacity
    """
    capex: pd.Series
    """
    Investment cost [$/1 unit of added capacity]
    """
    opex: pd.Series
    """
    Maintenance cost [$/1 unit of installed capacity]
    """
    min_capacity: pd.Series
    """
    Minimal amount of installed capacity of all units with this energy source type unit for a given year
    """
    max_capacity: pd.Series
    """
    Maximal amount of installed capacity of all units with this energy source type unit for a given year
    """
    min_capacity_increase: pd.Series
    """
    Maximal decrease of installed capacity of all units with this energy source type unit for a given year
    """
    max_capacity_increase: pd.Series
    """
    Minimal decrease of installed capacity of all units with this energy source type unit for a given year
    """
    tags: list[str] = field(default_factory=list)
    """
    Optional tag name list to group generator and storage types
    """

    def _validate_energy_source_type_base(
        self,
        network: Network,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        """
        Validates base energy source type parameters:
        - if attributes have proper type
        - if series attributes have defined data rows for every year in a simulation
        - if tags attribute is list of strings

        Args:
            exception_list (list[NetworkValidatorException]): list of exceptions to be raised
        """
        self._validate_name_type(exception_list)
        self._validate_tags(exception_list)

        for attr in ["life_time", "build_time"]:
            if not isinstance(getattr(self, attr), int):
                exception_list.append(
                    NetworkValidatorException(
                        f"Energy source {self.name} has invalid {attr}."
                        f" {attr.capitalize()} must be an integer, "
                        f"not {type(getattr(self, attr)).__name__}"
                    )
                )

        for attr in ["capex", "opex"]:
            validate_series(
                name=f"Energy source {self.name} {attr}",
                series=getattr(self, attr),
                length=network.constants.n_years,
                exception_list=exception_list,
            )

        for attr in [
            "min_capacity",
            "max_capacity",
            "min_capacity_increase",
            "max_capacity_increase",
        ]:
            series = getattr(self, attr)
            if validate_series(
                name=f"Energy source type {self.name} {attr}",
                series=series,
                length=network.constants.n_years,
                exception_list=exception_list,
                is_numeric=True,
            ) and not pd.isnull(series.iloc[0]):
                exception_list.append(
                    NetworkValidatorException(
                        f"Energy source type {self.name} {attr} must have a NaN value for the base year"
                    )
                )

    def _validate_tags(self, exception_list: list[NetworkValidatorException]) -> None:
        if not isinstance(self.tags, list) or any(
            not isinstance(t, str) for t in self.tags
        ):
            exception_list.append(
                NetworkValidatorException(
                    f"Energy source type {self.name} has invalid tags: {self.tags}. "
                )
            )
