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
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network_elements import EnergySourceType
from pyzefir.model.utils import validate_series
from pyzefir.utils.functions import is_flow_int

_logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pyzefir.model.network import Network


class SumNotEqualToOneWarning(Warning):
    pass


class GeneratorTypeValidatorExceptionGroup(NetworkValidatorExceptionGroup):
    pass


@dataclass(kw_only=True)
class GeneratorType(EnergySourceType):
    """
    A class that represents the GeneratorType in the network structure which stores parameters
    defined for a given type of generators
    """

    efficiency: pd.DataFrame
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
    power_utilization: pd.Series
    """
    Determines the percentage of the installed generator's rated power that
    can be used
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
    """Percentage difference between generations
    in relation to the installed capacity subsequent hours"""
    energy_curtailment_cost: dict[str, pd.Series] = field(default_factory=dict)
    """ energy curtailment for generator """
    generation_compensation: pd.Series | None = None
    """generation compensation parameters used to decrease objective
    pd.DataFrame with hours (rows) and years (columns)"""

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
        _logger.debug("Validating generator type object: %s...", self.name)
        exception_list: list[NetworkValidatorException] = []
        self._validate_energy_source_type_base(network, exception_list)
        if self.capacity_factor is not None and self.fuel is not None:
            exception_list.append(
                NetworkValidatorException(
                    "Generator type can have either capacity "
                    "factor or fuel at the same time"
                )
            )

        for name, gen_type in network.generator_types.items():
            curtailment_cost = gen_type.energy_curtailment_cost
            if len(curtailment_cost):
                self.validate_curtailment_cost(
                    network, name, curtailment_cost, exception_list
                )
        self._validate_fuels(exception_list, network)
        self._validate_capacity_factor(exception_list, network)
        self._validate_efficiency(exception_list, network)
        self._validate_emission_reduction(exception_list, network)
        self._validate_conversion_rate(exception_list, network)
        self._validate_power_utilization(network, exception_list)
        self._validate_generation_compensation(exception_list)

        if not isinstance(self.ramp, float | int):
            exception_list.append(
                NetworkValidatorException("Ramp value must be float or empty.")
            )

        elif not np.isnan(self.ramp) and not 0 < self.ramp < 1:
            exception_list.append(
                NetworkValidatorException(
                    f"Ramp value must be "
                    f"greater than 0 and less than 1, but it is {self.ramp}"
                )
            )

        if exception_list:
            _logger.debug("Got error validating network: %s", exception_list)
            raise GeneratorTypeValidatorExceptionGroup(
                f"While adding GeneratorType {self.name} following "
                f"errors occurred: ",
                exception_list,
            )
        _logger.debug("Generator type %s validation: Done", self.name)

    @property
    def inbound_energy_type(self) -> set[str]:
        """
        Gets set of energy types needed by the generator
        Returns:
            set[str]: Set of energy types.
        """
        return set(self.conversion_rate) if self.conversion_rate else set()

    def _validate_generation_compensation(
        self, exception_list: list[NetworkValidatorException]
    ) -> None:
        """
        Validation procedure checking:
        - Validates generation compensation type
        - Validates generation compensation data type

        Args:
            exception_list (NetworkValidatorException) - list of raised exceptions.

        Returns:
            None
        """
        if not isinstance(self.generation_compensation, pd.Series | None):
            exception_list.append(
                NetworkValidatorException(
                    f"Generation compensation of generator type {self.name} "
                    "must be type of pandas Series or None."
                )
            )
        elif (
            self.generation_compensation is not None
            and not pd.api.types.is_numeric_dtype(self.generation_compensation)
        ):
            exception_list.append(
                NetworkValidatorException(
                    f"Generation compensation of generator type {self.name} "
                    f"must contain float or int values only."
                )
            )

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
                    f" {type(self.fuel)} given"
                )
            )
        if self.fuel is not None and self.fuel not in network.fuels:
            exception_list.append(
                NetworkValidatorException(
                    f"Fuel {self.fuel} has not been added to the network"
                )
            )
        _logger.debug("Validate fuels: OK")

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
                    f"None or str type for capacity factor expected but type: "
                    f"{type(self.capacity_factor)} given"
                )
            )
        if (
            self.capacity_factor is not None
            and self.capacity_factor not in network.capacity_factors
        ):
            exception_list.append(
                NetworkValidatorException(
                    f"Capacity factor "
                    f"'{self.capacity_factor}' has not been added to the network"
                )
            )
        _logger.debug("Validate capacity factor: OK")

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
                NetworkValidatorException("Conversion rate cannot be None.")
            )
            return
        if not set(self.conversion_rate.keys()).issubset(network.energy_types):
            exception_list.append(
                NetworkValidatorException(
                    f"Conversion rate energy types do not exist "
                    f"in network energy types: {sorted(network.energy_types)}"
                )
            )
        _logger.debug("Validate conversion rate: OK")

    def _validate_efficiency(
        self, exception_list: list[NetworkValidatorException], network: Network
    ) -> None:
        """
        Validation procedure checking:
        - Validates if efficiency of generator type is not None
        - Validates whether efficiency energy types exist in network.energy_types
        - Validates if sum per each year is >= 1.0

        Args:
            exception_list (NetworkValidatorException) - list of raised exceptions.
            network (Network): network to which self is to be added.

        Returns:
            None
        """
        if self.efficiency is None:
            exception_list.append(
                NetworkValidatorException("Efficiency cannot be None.")
            )
            return
        if not set(self.efficiency.columns.to_list()).issubset(network.energy_types):
            exception_list.append(
                NetworkValidatorException(
                    f"Efficiency energy types do not exist "
                    f"in network energy types: {sorted(network.energy_types)}"
                )
            )
        is_sum_greater_than_or_1 = self.efficiency.sum(axis=1) - 1e-6 > 1
        hours_not_satisfying_condition = self.efficiency.index[
            is_sum_greater_than_or_1
        ].tolist()
        if hours_not_satisfying_condition:
            _logger.warning(
                "Generator type %s efficiency contains hours: %s which sum for each energy type is above 1",
                self.name,
                hours_not_satisfying_condition,
            )
            warnings.warn(
                f"Generator type {self.name} efficiency contains hours: {hours_not_satisfying_condition} "
                "which sum for each energy type is above 1",
                SumNotEqualToOneWarning,
            )

        _logger.debug("Validate efficiency: OK")

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
                NetworkValidatorException("Emission reduction cannot be None.")
            )
            return
        if emission_diff := set(self.emission_reduction.keys()).difference(
            network.emission_types
        ):
            exception_list.append(
                NetworkValidatorException(
                    f"Emission reduction emission types {emission_diff} do not exist "
                    f"in network emission types: {sorted(network.emission_types)}"
                )
            )
        _logger.debug("Validate emission reduction: OK")

    @staticmethod
    def _validate_curtailment_idx(
        network: Network,
        name: GeneratorType,
        curtailment_cost: pd.Series,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        if (
            len(curtailment_cost)
            and len(set(curtailment_cost.index)) != network.constants.n_years
        ):
            exception_list.append(
                NetworkValidatorException(
                    f"Incorrect year indices for energy curtailment cost of generator type "
                    f"<{str(name)}> The number of indexes should match the number of years"
                )
            )

    @staticmethod
    def _validate_curtailment_val(
        network: Network,
        name: GeneratorType,
        curtailment_cost: pd.Series,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        curtailment_cost_vals = curtailment_cost.values if len(curtailment_cost) else []
        vals = [el for el in curtailment_cost_vals if is_flow_int(el)]
        if len(vals) > 0 and len(vals) != network.constants.n_years:
            exception_list.append(
                NetworkValidatorException(
                    f"Incorrect values for energy curtailment cost of generator type <{str(name)}>"
                )
            )

    def validate_curtailment_cost(
        self,
        network: Network,
        name: GeneratorType,
        curtailment_cost: pd.Series,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        GeneratorType._validate_curtailment_idx(
            network, name, curtailment_cost, exception_list
        )
        GeneratorType._validate_curtailment_val(
            network, name, curtailment_cost, exception_list
        )

    def _validate_power_utilization(
        self,
        network: Network,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        """
        Validation procedure checking:
        - Validates if power utilization is instance of pd.Series
        - Validates if power utilization values range

        Args:
            exception_list (NetworkValidatorException) - list of raised exceptions.

        Returns:
            None
        """
        if validate_series(
            name="power_utilization",
            series=self.power_utilization,
            length=network.constants.n_hours,
            exception_list=exception_list,
        ):
            if not (incorrect_rows := self.power_utilization >= 0).all():
                incorrect_hours = incorrect_rows.index[~incorrect_rows].to_list()
                exception_list.append(
                    NetworkValidatorException(
                        f"Power utilization values must be greater "
                        f"or equal 0, but for hours: {incorrect_hours} it is not"
                    )
                )
        _logger.debug("Validate power utilization: OK")
