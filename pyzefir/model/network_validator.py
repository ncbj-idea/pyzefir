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
import logging
from abc import ABC, abstractmethod
from typing import Type

import numpy as np
import pandas as pd

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network import Network, NetworkElementsDict
from pyzefir.model.network_elements import AggregatedConsumer, Bus, Generator, Storage

_logger = logging.getLogger(__name__)


class BasicValidator(ABC):
    @staticmethod
    @abstractmethod
    def validate(
        network: Network, exception_list: list[NetworkValidatorException]
    ) -> None:
        pass


class NetworkValidator:
    def __init__(self, network: Network) -> None:
        self.network = network

    def validate(self) -> None:
        _logger.info("Validating network structure...")
        self._validate(
            RelativeEmissionLimitsValidation,
            BaseTotalEmissionValidation,
            BaseCapacityValidator,
            NetworkElementsValidation,
            PowerReserveValidation,
            DsrBusesOutValidation,
        )
        _logger.info("Network structure validation: Done.")

    def _validate(self, *validators: Type[BasicValidator]) -> None:
        exception_list: list[NetworkValidatorException] = []
        for validator in validators:
            validator.validate(self.network, exception_list)
        if exception_list:
            _logger.debug("Got error validating the network: %s", exception_list)
            raise NetworkValidatorExceptionGroup(
                "Following errors found during network validation: ", exception_list
            )


class DsrBusesOutValidation(BasicValidator):
    @staticmethod
    def validate(
        network: Network, exception_list: list[NetworkValidatorException]
    ) -> None:
        buses_out = [
            bus
            for lbs in network.local_balancing_stacks.values()
            for bus in lbs.buses_out.values()
        ]
        exception_list.extend(
            [
                NetworkValidatorException(
                    f"DSR {bus.dsr_type} could be added to 'out' buses only, "
                    f"but bus {bus.name} is not an 'out' bus."
                )
                for bus in network.buses.values()
                if bus.dsr_type and bus.name not in buses_out
            ]
        )


class PowerReserveValidation(BasicValidator):
    @staticmethod
    def validate(
        network: Network, exception_list: list[NetworkValidatorException]
    ) -> None:
        if PowerReserveValidation._validate_power_reserves_type(
            network.constants.power_reserves, exception_list
        ) and set(
            [
                key
                for value_dict in network.constants.power_reserves.values()
                for key in value_dict.keys()
            ]
        ):
            generators_with_tags = [
                network.generators[gen_name]
                for gen_name in network.generators
                if network.generators[gen_name].tags
            ]
            PowerReserveValidation._validate_power_reserves_tags(
                network, exception_list
            )
            PowerReserveValidation._validate_energy_type_matching(
                network, generators_with_tags, exception_list
            )
        _logger.debug("Power reserves are OK.")

    @staticmethod
    def _validate_power_reserves_type(
        power_reserves: dict[str, dict[str, float]],
        exception_list: list[NetworkValidatorException],
    ) -> bool:
        if not (
            isinstance(power_reserves, dict)
            and all(
                isinstance(k, str)
                and isinstance(internal_dict, dict)
                and all(
                    isinstance(kk, str) and isinstance(vv, (int, float))
                    for kk, vv in internal_dict.items()
                )
                for k, internal_dict in power_reserves.items()
            )
        ):
            exception_str = "Power reserve must be type of dict[str, dict[str, float]]."
            _logger.debug(exception_str)
            exception_list.append(NetworkValidatorException(exception_str))
            return False
        return True

    @staticmethod
    def _validate_energy_type_matching(
        network: Network,
        generators_with_tags: list[Generator],
        exception_list: list[NetworkValidatorException],
    ) -> None:
        for ee_type, tags in network.constants.power_reserves.items():
            for tag in tags:
                gen_with_tag = [gen for gen in generators_with_tags if tag in gen.tags]
                for gen in gen_with_tag:
                    if (
                        ee_type
                        not in network.generator_types[
                            gen.energy_source_type
                        ].energy_types
                    ):
                        exception_str = (
                            f"Generator: {gen.name} included in the tag: {tag} "
                            f"assigned to a given power reserve does not obtain "
                            f"the type of energy: {ee_type} that is assigned to the given power reserve."
                        )
                        _logger.debug(exception_str)
                        exception_list.append(NetworkValidatorException(exception_str))

    @staticmethod
    def _validate_power_reserves_tags(
        network: Network, exception_list: list[NetworkValidatorException]
    ) -> None:
        power_reserve_tags = set(
            [
                key
                for value_dict in network.constants.power_reserves.values()
                for key in value_dict.keys()
            ]
        )
        gen_tags = set(
            [
                tag
                for gen_name in network.generators.keys()
                for tag in network.generators[str(gen_name)].tags
            ]
        )
        if diff := sorted(power_reserve_tags.difference(gen_tags)):
            exception_str = (
                f"All tags assigned to a given power reserve must be defined and contain only generators,"
                f" but tags {diff} do not assign to generators, were missed or extra added."
            )
            _logger.debug(exception_str)
            exception_list.append(NetworkValidatorException(exception_str))


class RelativeEmissionLimitsValidation(BasicValidator):
    @staticmethod
    def validate(
        network: Network, exception_list: list[NetworkValidatorException]
    ) -> None:
        if RelativeEmissionLimitsValidation._validate_relative_emission_limits_types(
            network, exception_list
        ):
            RelativeEmissionLimitsValidation._validate_relative_emission_limits(
                network, exception_list
            )
        _logger.debug("Relative emission limits are OK.")

    @staticmethod
    def _validate_relative_emission_limits(
        network: Network, exception_list: list[NetworkValidatorException]
    ) -> None:
        if not all(
            [
                e_type in network.constants.relative_emission_limits.keys()
                for e_type in network.emission_types
            ]
        ):
            exception_list.append(
                NetworkValidatorException(
                    "Emission types in relative emission limits must correspond to the names "
                    "of the emission types listed in structure / Emission Types."
                )
            )
        for (
            e_type_name,
            series,
        ) in network.constants.relative_emission_limits.items():
            if not all([isinstance(v, int | float) for v in series.values]):
                exception_list.append(
                    NetworkValidatorException(
                        f"In each column corresponding to the emission type, "
                        f"the expected value type is float or int, "
                        f"but for {e_type_name} it's {[type(v) for v in series.values]}."
                    )
                )
            elif not series.dropna().between(0, 1, inclusive="both").all():
                exception_list.append(
                    NetworkValidatorException(
                        f"In each column corresponding to the emission type, "
                        f"the expected value must be in the range [0,1], "
                        f"but for {e_type_name} it's {str(series.values)}"
                    )
                )
            if len(series) and not pd.isnull(series.values[0]):
                exception_list.append(
                    NetworkValidatorException(
                        f"Year indices must start with 1 - you cannot specify "
                        f"a base year index with 0 in the column, "
                        f"but for {e_type_name} base index exists in input data."
                    )
                )

    @staticmethod
    def _validate_relative_emission_limits_types(
        network: Network, exception_list: list[NetworkValidatorException]
    ) -> bool:
        if not (
            isinstance(network.constants.relative_emission_limits, dict)
            and all(
                [
                    isinstance(key, str)
                    for key in network.constants.relative_emission_limits.keys()
                ]
            )
            and all(
                [
                    isinstance(data, pd.Series)
                    for data in network.constants.relative_emission_limits.values()
                ]
            )
        ):
            exception_str = (
                "Relative emission limits must be type of dict[str, pd.Series]."
            )
            _logger.debug(exception_str)
            exception_list.append(NetworkValidatorException(exception_str))
            return False
        return True


class BaseTotalEmissionValidation(BasicValidator):
    @staticmethod
    def validate(
        network: Network, exception_list: list[NetworkValidatorException]
    ) -> None:
        if not (
            isinstance(network.constants.base_total_emission, dict)
            and all(
                [
                    isinstance(key, str)
                    for key in network.constants.base_total_emission.keys()
                ]
            )
            and all(
                [
                    isinstance(v, float | int)
                    for v in network.constants.base_total_emission.values()
                ]
            )
        ):
            exception_str = (
                "Base total emission should be type of dict[str, float | int]."
            )
            _logger.debug(exception_str)
            exception_list.append(NetworkValidatorException(exception_str))
        _logger.debug("Base total emission is OK.")


class NetworkElementsValidation(BasicValidator):
    @staticmethod
    def validate(network: Network, exception_list: list[Exception]) -> None:
        network_elements_dicts = [
            attr
            for attr in vars(network).values()
            if isinstance(attr, NetworkElementsDict)
        ]
        for network_dict in network_elements_dicts:
            for element in network_dict.values():
                try:
                    element.validate(network)
                except Exception as e:
                    exception_list.append(e)
        _logger.debug("Network elements are OK.")


class BaseCapacityValidator(BasicValidator):
    @staticmethod
    def _match_unit_to_stack(
        unit_dict: NetworkElementsDict[Storage] | NetworkElementsDict[Generator],
        network_buses: NetworkElementsDict[Bus],
        bus_prop: str,
        stack_bus_map: dict[str, set[str]],
        exception_list: list[NetworkValidatorException],
    ) -> dict[str, str | None]:
        unit_stack_mapping = {}
        for unit in unit_dict.values():
            unit_buses = getattr(unit, bus_prop)
            buses = (
                {b for b in unit_buses if b in network_buses}
                if isinstance(unit_buses, set)
                else {unit_buses}
            )
            matching_stacks = set()
            for bus in buses:
                matched_stack = None
                for stack_name, bus_set in stack_bus_map.items():
                    if bus in bus_set:
                        matched_stack = stack_name
                        break
                matching_stacks.add(matched_stack)
            if len(matching_stacks) > 1:
                exception_str = f"Each generator ({unit.name}) must be used exactly in one or zero stacks."
                _logger.debug(exception_str)
                exception_list.append(NetworkValidatorException(exception_str))
            else:
                unit_stack_mapping[unit.name] = (
                    matching_stacks.pop() if matching_stacks else None
                )
        return unit_stack_mapping

    @staticmethod
    def _create_unit_to_stack_mapping(
        network: Network,
        exception_list: list[NetworkValidatorException],
    ) -> tuple[dict[str, str | None], dict[str, str | None]]:
        stack_buses = {
            stack.name: set().union(*stack.buses.values())
            for stack in network.local_balancing_stacks.values()
        }
        gen_stack_mapping = BaseCapacityValidator._match_unit_to_stack(
            unit_dict=network.generators,
            bus_prop="buses",
            network_buses=network.buses,
            stack_bus_map=stack_buses,
            exception_list=exception_list,
        )
        stor_stack_mapping = BaseCapacityValidator._match_unit_to_stack(
            unit_dict=network.storages,
            bus_prop="bus",
            network_buses=network.buses,
            stack_bus_map=stack_buses,
            exception_list=exception_list,
        )
        return gen_stack_mapping, stor_stack_mapping

    @staticmethod
    def _check_stack_base_cap(
        min_power: float | None,
        max_power: float | None,
        unit_name: str,
        aggr: AggregatedConsumer,
        base_fraction: float,
        base_cap: float,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        if min_power is None or max_power is None:
            exception_list.append(
                NetworkValidatorException(
                    f"For units ({unit_name}) that are used in a local balancing stack, "
                    "attributes min_device_power and max_device_power must be defined"
                )
            )
        else:
            n_consumer = aggr.n_consumers[0]
            if not (
                (
                    base_fraction * n_consumer * min_power <= base_cap
                    or np.isclose(base_fraction * n_consumer * min_power, base_cap)
                )
                and (
                    base_cap <= base_fraction * n_consumer * max_power
                    or np.isclose(base_cap, base_fraction * n_consumer * max_power)
                )
            ):
                exception_list.append(
                    NetworkValidatorException(
                        f"In energy source {unit_name}, if base capacity has been defined, "
                        f"the compound inequality must be true with numerical tolerance: "
                        f"base_fraction * n_consumers * min_device_nom_power <= base_capacity <= "
                        f"base_fraction * n_consumers * max_device_nom_power, but it is "
                        f"{base_fraction} * {n_consumer} * {min_power} "
                        f"({base_fraction * n_consumer * min_power}) <= {base_cap} "
                        f"<= {base_fraction} * {n_consumer} * {max_power} "
                        f"({base_fraction * n_consumer * max_power}) instead"
                    )
                )

    @staticmethod
    def _check_base_cap(
        unit_stack_mapping: dict[str, str | None],
        unit_dict: NetworkElementsDict[Generator] | NetworkElementsDict[Storage],
        aggregated_consumers: NetworkElementsDict[AggregatedConsumer],
        stack_aggr_map: dict[str, str],
        exception_list: list[NetworkValidatorException],
    ) -> None:
        for unit_name, stack in unit_stack_mapping.items():
            unit = unit_dict[unit_name]
            max_power = unit.max_device_nom_power
            min_power = unit.min_device_nom_power
            base_cap = unit.unit_base_cap
            if base_cap is None:
                exception_list.append(
                    NetworkValidatorException(
                        f"Base capacity for unit {unit_name} is not defined"
                    )
                )
            elif stack is not None:
                aggr = aggregated_consumers[stack_aggr_map[stack]]
                base_fraction = aggr.stack_base_fraction[stack]
                BaseCapacityValidator._check_stack_base_cap(
                    min_power=min_power,
                    max_power=max_power,
                    unit_name=unit_name,
                    aggr=aggr,
                    base_fraction=base_fraction,
                    base_cap=base_cap,
                    exception_list=exception_list,
                )
            else:
                if min_power is not None or max_power is not None:
                    exception_list.append(
                        NetworkValidatorException(
                            f"For units ({unit_name}) that are not used in a local balancing stack, "
                            "attributes min_device_power and max_device_power must not be defined"
                        )
                    )

    @staticmethod
    def _create_stack_aggr_map(
        aggregated_consumers: NetworkElementsDict[AggregatedConsumer],
        exception_list: list[NetworkValidatorException],
    ) -> dict[str, str]:
        stack_aggr_map = {}
        for aggr in aggregated_consumers.values():
            for stack in aggr.available_stacks:
                if stack in stack_aggr_map:
                    exception_str = f"Stack {stack} is connected to more than one aggregated consumer"
                    _logger.debug(exception_str)
                    exception_list.append(NetworkValidatorException(exception_str))
                stack_aggr_map[stack] = aggr.name
        return stack_aggr_map

    @staticmethod
    def validate(
        network: Network, exception_list: list[NetworkValidatorException]
    ) -> None:
        (
            gen_stack_mapping,
            stor_stack_mapping,
        ) = BaseCapacityValidator._create_unit_to_stack_mapping(network, exception_list)
        stack_aggr_map = BaseCapacityValidator._create_stack_aggr_map(
            network.aggregated_consumers, exception_list
        )
        BaseCapacityValidator._check_base_cap(
            unit_stack_mapping=gen_stack_mapping,
            unit_dict=network.generators,
            aggregated_consumers=network.aggregated_consumers,
            stack_aggr_map=stack_aggr_map,
            exception_list=exception_list,
        )
        BaseCapacityValidator._check_base_cap(
            unit_stack_mapping=stor_stack_mapping,
            unit_dict=network.storages,
            aggregated_consumers=network.aggregated_consumers,
            stack_aggr_map=stack_aggr_map,
            exception_list=exception_list,
        )


class GeneratorCapacityCostValidator(BasicValidator):
    @staticmethod
    def validate(
        network: Network, exception_list: list[NetworkValidatorException]
    ) -> None:
        """
        If generator_capacity_cost is set to 'netto', all generator_types must have exactly
        one energy type, if it is set to 'brutto' - no restrictions.
        """
        if network.constants.generator_capacity_cost == "netto":
            for generator_type in network.generator_types.values():
                if len(generator_type.energy_types) > 1:
                    exception_list.append(
                        NetworkValidatorException(
                            f"generator type '{generator_type.name}' have more than one energy "
                            f"type defined, but generator_capacity_cost "
                            f"parameter is set to '{network.constants.generator_capacity_cost}'; if you want to "
                            "have generator types with more than one energy type, please set generator_capacity_cost "
                            "to 'brutto'"
                        )
                    )
