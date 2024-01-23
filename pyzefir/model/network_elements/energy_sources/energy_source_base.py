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
class EnergySource(NetworkElement, ABC):
    """
    A class holding individual parameters for a given element
    """

    energy_source_type: str
    """
    Unique name of energy source element type
    """
    unit_base_cap: float
    """
    Installed capacity of a given energy source in the initial year (starting condition of the optimization)
    """
    unit_min_capacity: pd.Series
    """
    Minimal amount of installed capacity of given energy source for a given year
    """
    unit_max_capacity: pd.Series
    """
    Maximal amount of installed capacity of given energy source for a given year
    """
    unit_min_capacity_increase: pd.Series
    """
    Maximal decrease of installed capacity of given energy source for a given year
    """
    unit_max_capacity_increase: pd.Series
    """
    Minimal decrease of installed capacity of given energy source for a given year
    """
    min_device_nom_power: float | None = None
    """
    Minimal device nominal power for single device
    """
    max_device_nom_power: float | None = None
    """
    Maximum device nominal power for single device
    """
    tags: list[str] = field(default_factory=list)
    """
    Optional tag name list to group generators and storages
    """

    def _validate_device_nominal_power(
        self,
        nom_power_name: str,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        nom_power = getattr(self, nom_power_name)
        if not isinstance(nom_power, int | float | None):
            exception_list.append(
                NetworkValidatorException(
                    f"Energy source {self.name} has invalid {nom_power_name}. "
                    f"{nom_power_name} must be an instance of one of the types: float, int or None, "
                    f"not {type(nom_power).__name__}"
                )
            )
        elif nom_power is not None and not nom_power >= 0:
            exception_list.append(
                NetworkValidatorException(
                    f"{nom_power_name} for {self.name} has invalid value. "
                    f"It must be greater or equal to zero, but it is: {nom_power}"
                )
            )

    def _validate_base_energy_source(
        self, network: Network, exception_list: list[NetworkValidatorException]
    ) -> None:
        """
        Validates base energy source parameters:
        - if energy_source_type is proper string
        - if unit_base_cap is numeric
        - if tags attribute is list of strings

        Method validate runs following validate methods:
        - _validate_device_nominal_power

        Args:
            network (Network): network to which self is to be added

        Raises:
            NetworkValidatorExceptionGroup: If exception_list contains exception.
        """
        self._validate_name_type(exception_list)
        if not isinstance(self.energy_source_type, str):
            exception_list.append(
                NetworkValidatorException(
                    f"Energy source {self.name} has invalid energy source type."
                    " Energy source type must be a string, "
                    f"not {type(self.energy_source_type).__name__}"
                )
            )

        if not isinstance(self.unit_base_cap, (int, float)):
            exception_list.append(
                NetworkValidatorException(
                    f"Energy source {self.name} has invalid unit base capacity. "
                    "Unit base capacity must be numeric, "
                    f"not {type(self.unit_base_cap).__name__}"
                )
            )

        if not isinstance(self.tags, list) or any(
            not isinstance(t, str) for t in self.tags
        ):
            exception_list.append(
                NetworkValidatorException(
                    f"Energy source {self.name} has invalid tags: {self.tags}. "
                )
            )

        for attr in [
            "unit_min_capacity",
            "unit_max_capacity",
            "unit_min_capacity_increase",
            "unit_max_capacity_increase",
        ]:
            series = getattr(self, attr)
            if validate_series(
                name=f"Energy source {self.name} {attr}",
                series=series,
                length=network.constants.n_years,
                exception_list=exception_list,
                is_numeric=True,
            ) and not pd.isnull(series.iloc[0]):
                exception_list.append(
                    NetworkValidatorException(
                        f"Energy source {self.name} {attr} must have a NaN value for the base year"
                    )
                )

        self._validate_device_nominal_power("min_device_nom_power", exception_list)
        self._validate_device_nominal_power("max_device_nom_power", exception_list)
