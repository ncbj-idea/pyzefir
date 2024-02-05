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

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network_element import NetworkElement
from pyzefir.model.utils import check_interval, validate_dict_type, validate_series

if TYPE_CHECKING:
    from pyzefir.model.network import Network


@dataclass
class AggregatedConsumer(NetworkElement):
    """
    A class that represents the AggregatedConsumer element in the
    network structure which aggregate LocalBalancingStack
    """

    demand_profile: str
    """
    Unique name of DemandProfile which holds normalized demand profile pd.Series
    """
    stack_base_fraction: dict[str, float]
    """
    Dictionary mapping LocalBalancingStack.name to float representing fraction
    of aggregated consumer in 0 year for each technology.
    """
    yearly_energy_usage: dict[str, pd.Series]
    """
    Total amount of energy (all types) that needs to be provided to the given
    aggregate in a given year
    """
    min_fraction: dict[str, pd.Series]
    """
    Dictionary mapping LocalBalancingStack.name to float representing minimal fraction
    of aggregated consumer using energy from corresponding LocalBalancingStack.
    Fraction value must be in range <0;1> and sum to 1 for specific aggregate.
    """
    max_fraction: dict[str, pd.Series]
    """
    Dictionary mapping LocalBalancingStack.name to float representing maximal fraction
    of aggregated consumer using energy from corresponding LocalBalancingStack.
    Fraction value must be in range <0;1> and sum to 1 for specific aggregate.
    """
    max_fraction_decrease: dict[str, pd.Series]
    """
    Dictionary mapping LocalBalancingStack.name to float representing maximal fraction decrease
    of aggregated consumer using energy from corresponding LocalBalancingStack.
    Fraction value must be in range <0;1> and sum to 1 for specific aggregate.
    """
    max_fraction_increase: dict[str, pd.Series]
    """
    Dictionary mapping LocalBalancingStack.name to float representing maximal fraction increase
    of aggregated consumer using energy from corresponding LocalBalancingStack.
    Fraction value must be in range <0;1> and sum to 1 for specific aggregate.
    """
    n_consumers: pd.Series
    """
    Number of consumers represented by the aggregate
    """
    average_area: float | None
    """
    Average area of this aggregate
    """

    @property
    def available_stacks(self) -> list[str]:
        """
        Get a unique list of all LocalBalancingStack object names mapped to the
        object via stack_base_fraction parameter

        Returns:
            list[str]: List of LocalBalancingStack names.
        """
        return list(self.stack_base_fraction.keys())

    def _validate_demand_profile(
        self, network: Network, exception_list: list[NetworkValidatorException]
    ) -> None:
        """
        Validate demand profile of AggregatedConsumer.
            - Validates if demand profile is a string
            - Validates if demand profile is defined in the network

        Args:
            network (Network): Network to which AggregatedConsumer is
                to be added.
            exception_list (list[NetworkValidatorException]): List of
                exceptions to be raised.
        """
        if not isinstance(self.demand_profile, str):
            exception_list.append(
                NetworkValidatorException(
                    f"Demand for AggregatedConsumer {self.name} "
                    "must be given as a string"
                )
            )
            return  # do not need validate if demand_profile is not str
        if network.demand_profiles.get(self.demand_profile) is None:
            exception_list.append(
                NetworkValidatorException(
                    f"Demand {self.demand_profile} for AggregatedConsumer {self.name} "
                    "must be defined in the network"
                )
            )
        else:
            demand_profile_energy_types = [
                *network.demand_profiles[self.demand_profile].normalized_profile.keys()
            ]

            for stack_name in self.available_stacks:
                stack = network.local_balancing_stacks.get(stack_name)
                if stack is None:
                    continue
                stack_energy_types = set(stack.buses_out.keys())
                if diff := set(stack_energy_types).symmetric_difference(
                    demand_profile_energy_types
                ):
                    exception_list.append(
                        NetworkValidatorException(
                            f"Energy types of aggregated consumer demand profile {self.name} "
                            f"are different than energy types defined in the connected stack {stack_name}. "
                            f"Difference: {diff}"
                        )
                    )

    def _validate_stack_base_fraction_type(
        self, exception_list: list[NetworkValidatorException]
    ) -> bool:
        """
        Validate type of stack_base_fraction parameter.
            - Validates if stack_base_fraction is a dictionary

        Args:
            exception_list (list[NetworkValidatorException]): List of
                exceptions to be raised.

        Returns:
            bool: Returns True if the validation is successful, indicating that the
            stack_base_fraction parameter has the correct type and format.
            Returns False if validation fails.
        """

        if not isinstance(self.stack_base_fraction, dict):
            exception_list.append(
                NetworkValidatorException(
                    f"Stack base fractions for AggregatedConsumer {self.name} "
                    "must be given as a dictionary"
                )
            )
            return False

        is_validation_ok = True

        if not all(
            isinstance(fraction, (int, float))
            for fraction in self.stack_base_fraction.values()
        ):
            exception_list.append(
                NetworkValidatorException(
                    f"Stack base fractions for AggregatedConsumer {self.name} "
                    "must be given as a dictionary with values of type float"
                )
            )
            is_validation_ok = False

        if not all(isinstance(key, str) for key in self.stack_base_fraction.keys()):
            exception_list.append(
                NetworkValidatorException(
                    f"Stack base fractions for AggregatedConsumer {self.name} "
                    "must be given as a dictionary with keys of type str"
                )
            )
            is_validation_ok = False

        return is_validation_ok

    def _validate_stack_base_fraction(
        self, network: Network, exception_list: list[NetworkValidatorException]
    ) -> None:
        """
        Validate LocalBalancingStacks in aggregated consumer.
            - Validate type
            - Validates if LocalBalancingStacks are defined in the network
            - Validates if sum of fractions is equal to 1
            - Validates if fractions are in range <0;1>

        Args:
            network (Network): Network to which AggregatedConsumer is to
                be added.
            exception_list (list[NetworkValidatorException]): List of exceptions
                to be raised.
        """
        if not self._validate_stack_base_fraction_type(exception_list):
            return  # skip validation if incorrect types

        if not math.isclose(1, fraction_sum := sum(self.stack_base_fraction.values())):
            exception_list.append(
                NetworkValidatorException(
                    "Local balancing stack fractions for aggregated consumer "
                    f"{self.name} do not sum to 1, "
                    f"but to {fraction_sum} instead"
                )
            )

        for stack, fraction in self.stack_base_fraction.items():
            if not check_interval(lower_bound=0, upper_bound=1, value=fraction):
                exception_list.append(
                    NetworkValidatorException(
                        f"The value of the {stack} is inconsistent with th expected bounds of "
                        f"the interval: 0 <= {fraction} <= 1"
                    )
                )

            if stack not in network.local_balancing_stacks:
                exception_list.append(
                    NetworkValidatorException(
                        f"Local balancing stack {stack} available for "
                        "aggregated consumer "
                        f"{self.name} does not exist in the network"
                    )
                )
            if network.constants.binary_fraction and fraction not in [0, 1]:
                exception_list.append(
                    NetworkValidatorException(
                        f"For binary fraction setting, stack base fraction must contain only values 0 and 1. "
                        f"Found value: {fraction}"
                    )
                )

    def _validate_yearly_energy_usage_types(
        self, exception_list: list[NetworkValidatorException]
    ) -> bool:
        """
        Validate types of yearly_energy_usage parameter.

        Args:
            exception_list (list[NetworkValidatorException]): List of exceptions
                to be raised.

        Returns:
            bool: Returns True if the validation is successful, indicating that the
            yearly_energy_usage parameter has the correct type and format.
            Returns False if validation fails.
        """
        if not isinstance(self.yearly_energy_usage, dict):
            exception_list.append(
                NetworkValidatorException("Yearly energy usage must be of dict type")
            )
            return False

        is_validation_ok = True
        if not all(
            isinstance(energy_source, str)
            for energy_source in self.yearly_energy_usage.keys()
        ):
            exception_list.append(
                NetworkValidatorException("Energy type must be of str type")
            )
            is_validation_ok = False

        if not all(
            isinstance(energy_usage, pd.Series)
            for energy_usage in self.yearly_energy_usage.values()
        ):
            exception_list.append(
                NetworkValidatorException("Energy usage must be of pd.Series type")
            )
            is_validation_ok = False

        return is_validation_ok

    def _validate_yearly_energy_usage(
        self,
        network: Network,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        """
        Validate yearly energy usage series of AggregatedConsumer element.
            - Validate type
            - Validate that all energy types have same index
            - Validate that all energy types are defined in the network

        Args:
            exception_list (list[NetworkValidatorException]): List of exceptions
                to be raised.
        """

        if not self._validate_yearly_energy_usage_types(exception_list):
            return  # Do not validate further if types are invalid

        if len(self.yearly_energy_usage) == 0:
            exception_list.append(
                NetworkValidatorException(
                    "Yearly energy usage must contain at least one energy type"
                )
            )
            return

        keys = set(self.yearly_energy_usage.keys())
        index = self.yearly_energy_usage[next(iter(keys))].index

        if not all(self.yearly_energy_usage[key].index.equals(index) for key in keys):
            exception_list.append(
                NetworkValidatorException(
                    "Yearly energy usage series for aggregated consumer "
                    f"{self.name} must have the same index"
                )
            )

        for stack_name in self.available_stacks:
            stack = network.local_balancing_stacks.get(stack_name)
            if stack is None:
                continue
            stack_energy_types = set(stack.buses_out.keys())
            if diff := set(stack_energy_types).symmetric_difference(keys):
                exception_list.append(
                    NetworkValidatorException(
                        f"Energy types of aggregated consumer {self.name} "
                        f"are different than energy types defined in the connected stack {stack_name}. "
                        f"Difference: {diff}"
                    )
                )

    def _validate_fraction_types(
        self,
        number_of_years: int,
        exception_list: list[NetworkValidatorException],
        fraction_name: str,
    ) -> bool:
        """
        Validate type of fraction parameter.
            - Validates if fraction is a dictionary
            - Values of dictionary is pd.Series
            - pd.Series index: int and values: float

        Args:
            exception_list (list[NetworkValidatorException]): List of
                exceptions to be raised.
            number_of_years (int): The length of the simulation period

        Returns:
            bool: Returns True if the validation is successful, indicating that the
            fraction parameter has the correct type and format.
            Returns False if validation fails.
        """
        if validate_dict_type(
            dict_to_validate=getattr(self, fraction_name),
            key_type=str,
            value_type=pd.Series,
            parameter_name=f"fraction {fraction_name} for {self.name}",
            key_parameter_name="aggregate",
            value_parameter_name="fraction_series",
            exception_list=exception_list,
        ):
            return self._validate_series_types(
                number_of_years, exception_list, fraction_name
            )
        else:
            return False

    def _validate_series_types(
        self,
        number_of_years: int,
        exception_list: list[NetworkValidatorException],
        fraction_name: str,
    ) -> bool:
        """
        Validate fraction series structure types.
            - series index are int type
            - series values are float type (float or np.nan)
        Args:
            exception_list (list[NetworkValidatorException]): List of
                exceptions to be raised.
            number_of_years (int): The length of the simulation period
        Returns:
            bool: Returns True if the validation is successful, indicating that the
            fraction parameter has the correct type and format.
            Returns False if validation fails.
        """
        is_validation_ok = True
        for stack, series in getattr(self, fraction_name).items():
            if validate_series(
                name=f"Local balancing stack {stack} fraction {fraction_name} series",
                series=series,
                length=number_of_years,
                exception_list=exception_list,
                index_type=np.integer,
                values_type=np.floating,
            ):
                values_outside_range = series.loc[~series.between(0, 1)].dropna()
                if not values_outside_range.empty:
                    exception_list.append(
                        NetworkValidatorException(
                            f"Fraction {fraction_name} in LBS {stack} for AggregatedConsumer "
                            f"{self.name} values must be given in range"
                            f" <0:1> but {values_outside_range.tolist()} given instead"
                        )
                    )
                    is_validation_ok = False
            else:
                is_validation_ok = False
        return is_validation_ok

    def _validate_fraction(
        self,
        fraction_name: str,
        network: Network,
        exception_list: list[NetworkValidatorException],
    ) -> bool:
        is_fraction_ok = True
        fraction = getattr(self, fraction_name)
        if not self._validate_fraction_types(
            network.constants.n_years, exception_list, fraction_name
        ):
            return False
        for stack, fraction_series in fraction.items():
            if stack not in self.available_stacks:
                exception_list.append(
                    NetworkValidatorException(
                        f"Local balancing stack {stack} given for fraction "
                        f"{fraction_name} is not defined in base_fractions"
                    )
                )
                is_fraction_ok = False
            if not pd.isna(fraction_series.iloc[0]):
                exception_list.append(
                    NetworkValidatorException(
                        f"Local balancing stack {stack} in fraction attribute {fraction_name} "
                        "detected value for base year. This attribute could be provided "
                        "for every year except the base year."
                    )
                )
                is_fraction_ok = False
            if network.constants.binary_fraction:
                wrong_indices = fraction_series[
                    (~fraction_series.isin([0, 1, np.nan]))
                ].index.to_list()
                if wrong_indices:
                    exception_list.append(
                        NetworkValidatorException(
                            f"For binary fraction setting, {fraction_name} in stack {stack} must contain "
                            f"only values 0 or 1 for all years. "
                            f"Detected incorrect values for years: {wrong_indices}. "
                            f"Values: {fraction_series[wrong_indices].to_list()}"
                        )
                    )
                    is_fraction_ok = False
        return is_fraction_ok

    def _validate_fractions(
        self, network: Network, exception_list: list[NetworkValidatorException]
    ) -> None:
        """
        Validates the fraction data for the aggregated consumer in the network
        - Validates fraction types
        - Checks if local balancing stacks exist in the network
        - Compares the length of fraction series with the number of years in the model
        - Validates summed series

        Args:
            network (Network): The network containing the consumer and its related data.
            exception_list (list[NetworkValidatorException]): A list to collect validation exceptions.
        """
        all_fractions_ok = True
        fraction_attr_names = [
            "min_fraction",
            "max_fraction",
            "max_fraction_decrease",
            "max_fraction_increase",
        ]
        for fraction_name in fraction_attr_names:
            all_fractions_ok = (
                self._validate_fraction(fraction_name, network, exception_list)
                and all_fractions_ok
            )

        if all_fractions_ok:
            self._validate_fraction_values(exception_list)

    def _validate_fraction_aggregated_values(
        self,
        min_df: pd.DataFrame,
        max_df: pd.DataFrame,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        min_sum = min_df.sum(axis=1, skipna=False)
        max_sum = max_df.sum(axis=1, skipna=False)
        if (min_df > 1).any().any():
            exception_list.append(
                NetworkValidatorException(
                    f"Minimal fraction must be lower or equal to 1 for each year. "
                    f"Detected values greater than 1 for years: {min_df.index[(min_df > 1).any(axis=1)].to_list()}"
                )
            )

        if (series := ((min_sum > 1) & (~min_sum.isna()))).any():
            exception_list.append(
                NetworkValidatorException(
                    f"Minimal fraction sum must be lower than 1. "
                    f"Detected values greater than 1 for years: {min_df.index[series].to_list()}"
                )
            )
        if (
            set(self.available_stacks) == set(max_df.columns)
            and (series := ((max_sum < 1) & (~max_sum.isna()))).any()
        ):
            exception_list.append(
                NetworkValidatorException(
                    f"If maximal fraction is defined for every available stack in this aggregate consumer, "
                    f"then sum of these fractions must not be lower than 1. "
                    f"Detected incorrect values for years: {max_df.index[series].to_list()}"
                )
            )

    def _validate_fraction_values(
        self,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        min_df = pd.DataFrame(self.min_fraction)
        max_df = pd.DataFrame(self.max_fraction)
        self._validate_fraction_aggregated_values(min_df, max_df, exception_list)
        self._validate_stack_fractions(exception_list)

    def _validate_stack_fractions(
        self,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        for stack in self.available_stacks:
            min_fraction = self.min_fraction.get(stack)
            max_fraction = self.max_fraction.get(stack)
            max_fraction_increase = self.max_fraction_increase.get(stack)
            max_fraction_decrease = self.max_fraction_decrease.get(stack)
            self._validate_fraction_per_stack(
                min_fraction,
                max_fraction,
                max_fraction_increase,
                max_fraction_decrease,
                exception_list,
            )

    @staticmethod
    def _validate_fraction_per_stack(
        min_fraction: pd.Series | None,
        max_fraction: pd.Series | None,
        max_fraction_increase: pd.Series | None,
        max_fraction_decrease: pd.Series | None,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        if min_fraction is None or max_fraction is None:
            return
        if (series := (min_fraction > max_fraction)).any():
            exception_list.append(
                NetworkValidatorException(
                    f"Maximal fraction must be greater or equal than minimal fraction value. "
                    f"Detected incorrect values for years: {min_fraction.index[series].to_list()}"
                )
            )
        if max_fraction_increase is not None:
            increase = min_fraction.shift(-1) - max_fraction
            if not (
                series := (
                    (increase <= max_fraction_increase)
                    | increase.isnull()
                    | max_fraction_increase.isnull()
                )
            ).all():
                exception_list.append(
                    NetworkValidatorException(
                        f"In every year minimal fraction in the next year must be "
                        f"lower than maximal fraction plus maximal fraction increase. "
                        f"Detected incorrect values for years: {max_fraction_increase.index[~series].to_list()}"
                    )
                )
        if max_fraction_decrease is not None:
            decrease = min_fraction - max_fraction.shift(-1)
            if not (
                series := (
                    (decrease <= max_fraction_decrease)
                    | decrease.isnull()
                    | max_fraction_decrease.isnull()
                )
            ).all():
                exception_list.append(
                    NetworkValidatorException(
                        f"In every year maximal fraction in the next year must be "
                        f"lower than minimal fraction plus maximal fraction decrease. "
                        f"Detected incorrect values for years: {max_fraction_decrease.index[~series].to_list()}"
                    )
                )

    def _validate_n_consumers(
        self,
        n_years: int,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        validate_series(
            name="N_consumer series",
            series=self.n_consumers,
            length=n_years,
            exception_list=exception_list,
            index_type=np.integer,
            values_type=np.integer,
            allow_null=False,
        )
        if (self.n_consumers <= 0).any():
            exception_list.append(
                NetworkValidatorException(
                    "For n_consumers series all values must be positive and given for each year in simulation."
                )
            )

    def validate(self, network: Network) -> None:
        """
        Validate AggregatedConsumer element.
            - Validate demand profile
            - Validate stack base fraction
            - Validate yearly energy usage
            - Validate fraction

        Args:
            network (Network): Network to which AggregatedConsumer is
                to be added.

        Raises:
            NetworkValidatorExceptionGroup: If AggregatedConsumer is invalid.
        """
        exception_list: list[NetworkValidatorException] = []

        self._validate_name_type(exception_list=exception_list)
        self._validate_demand_profile(network=network, exception_list=exception_list)
        self._validate_yearly_energy_usage(
            network=network, exception_list=exception_list
        )
        self._validate_n_consumers(
            n_years=network.constants.n_years, exception_list=exception_list
        )

        self._validate_stack_base_fraction(
            network=network, exception_list=exception_list
        )
        self._validate_fractions(network=network, exception_list=exception_list)
        if self.average_area is not None:
            if not isinstance(self.average_area, float):
                exception_list.append(
                    NetworkValidatorException(
                        f"Average area for AggregatedConsumer {self.name} must be given as float"
                    )
                )

        if exception_list:
            raise NetworkValidatorExceptionGroup(
                f"While adding AggregatedConsumer {self.name} following errors occurred: ",
                exception_list,
            )
