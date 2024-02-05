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

import math
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


class BasicValidator(ABC):
    @staticmethod
    @abstractmethod
    def validate(
        network: Network, exception_list: list[NetworkValidatorException]
    ) -> None:
        ...


class NetworkValidator:
    def __init__(self, network: Network) -> None:
        self.network = network

    def validate(self) -> None:
        self._validate(
            RelativeEmissionLimitsValidation,
            BaseTotalEmissionValidation,
            BaseCapacityValidator,
            NetworkElementsValidation,
            NetworkGenerationFraction,
            PowerReserveValidation,
        )

    def _validate(self, *validators: Type[BasicValidator]) -> None:
        exception_list: list[NetworkValidatorException] = []
        for validator in validators:
            validator.validate(self.network, exception_list)
        if exception_list:
            raise NetworkValidatorExceptionGroup(
                "Following errors found during network validation: ", exception_list
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
            exception_list.append(
                NetworkValidatorException(
                    "Power reserve must be type of dict[str, dict[str, float]]."
                )
            )
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
                        exception_list.append(
                            NetworkValidatorException(
                                f"Generator: {gen.name} included in the tag: {tag} "
                                f"assigned to a given power reserve does not obtain "
                                f"the type of energy: {ee_type} that is assigned to the given power reserve."
                            )
                        )

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
        if diff := sorted(power_reserve_tags.symmetric_difference(gen_tags)):
            exception_list.append(
                NetworkValidatorException(
                    f"All tags assigned to a given power reserve must be defined and contain only generators, "
                    f"but tags {diff} do not assign to generators, were missed or extra added."
                )
            )


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
            exception_list.append(
                NetworkValidatorException(
                    "Relative emission limits must be type of dict[str, pd.Series]."
                )
            )
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
            exception_list.append(
                NetworkValidatorException(
                    "Base total emission should be type of dict[str, float | int].",
                )
            )


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
                exception_list.append(
                    NetworkValidatorException(
                        f"Each generator ({unit.name}) must be used exactly in one or zero stacks."
                    )
                )
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
                    or math.isclose(base_fraction * n_consumer * min_power, base_cap)
                )
                and (
                    base_cap <= base_fraction * n_consumer * max_power
                    or math.isclose(base_cap, base_fraction * n_consumer * max_power)
                )
            ):
                exception_list.append(
                    NetworkValidatorException(
                        f"In energy source {unit_name}, if base capacity has been defined, "
                        f"the compound inequality must be true: "
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
                    exception_list.append(
                        NetworkValidatorException(
                            f"Stack {stack} is connected to more than one aggregated consumer"
                        )
                    )
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


class NetworkGenerationFraction(BasicValidator):
    @staticmethod
    def _validate_min_max_gen_frac(
        min_max: dict[str, dict[tuple[str, str], float]],
        message: str,
        energy_types: list[str],
        exception_list: list[NetworkValidatorException],
    ) -> None:
        """Check if all min/max generation fraction are in [0,1] and if energy type is correct"""

        for energy_type, tags_gen_frac in min_max.items():
            for tags_pair, generation_fraction in tags_gen_frac.items():
                if energy_type not in energy_types:
                    exception_list.append(
                        NetworkValidatorException(
                            f"Incorrect energy type <{energy_type}> for tags <{tags_pair}>"
                        )
                    )
                if (
                    generation_fraction < 0
                    or generation_fraction > 1
                    or np.isnan(generation_fraction)
                ):
                    exception_list.append(
                        NetworkValidatorException(
                            f"{message} generation fraction <{generation_fraction}> for energy type <{energy_type}>, "
                            f"tags <{tags_pair}> "
                            f"must be a number greater than zero and "
                            f"smaller than one "
                        )
                    )

    @staticmethod
    def _get_n_tag_elemets(
        tag: str, subtag: str, network: Network
    ) -> tuple[int, int, int, int]:
        gen, stor = network.generators, network.storages
        n_gen_tag = len([v.tags for k, v in gen.items() if tag in v.tags])
        n_stor_tag = len([v.tags for k, v in stor.items() if tag in v.tags])
        n_gen_subtag = len([v.tags for k, v in gen.items() if subtag in v.tags])
        n_stor_subtag = len([v.tags for k, v in stor.items() if subtag in v.tags])
        return n_gen_tag, n_stor_tag, n_gen_subtag, n_stor_subtag

    @staticmethod
    def _validate_proper_subtags(
        min_max: dict[str, dict[tuple[str, str], float]],
        network: Network,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        """Check subtag is a proper set"""
        for energy_type, tags_gen_frac in min_max.items():
            for tags in tags_gen_frac.keys():
                tag, subtag = tags
                (
                    n_gen_tag,
                    n_stor_tag,
                    n_gen_subtag,
                    n_stor_subtag,
                ) = NetworkGenerationFraction._get_n_tag_elemets(tag, subtag, network)
                if n_gen_tag + n_stor_tag <= n_gen_subtag + n_stor_subtag:
                    exception_list.append(
                        NetworkValidatorException(
                            f"Subtag <{subtag}> is not a proper subset of the tag set <{tag}>"
                        )
                    )

    @staticmethod
    def _gen_et_validation(
        energy_types_dict: dict[str, str],
        network: Network,
        tag: str,
        energy_type: str,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        gen = network.generators
        gens_et = [
            {
                "name": v.name,
                "energy_type": {
                    energy_types_dict[bus]
                    for bus in v.buses.union({v.bus})
                    if bus is not None
                },
            }
            for k, v in gen.items()
            if tag in v.tags
        ]
        for gen_et in gens_et:
            if energy_type not in gen_et["energy_type"]:
                unit_name = gen_et["name"]
                exception_list.append(
                    NetworkValidatorException(
                        f"Energy type for generators of the tag: {tag} "
                        f"for <{unit_name}> do not match energy type in Generation Fraction"
                    )
                )

    @staticmethod
    def _stor_et_validation(
        energy_types_dict: dict[str, str],
        network: Network,
        tag: str,
        energy_type: str,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        stor = network.storages
        stors_et = [
            {
                "name": v.name,
                "energy_type": {
                    energy_types_dict[bus] for bus in {v.bus} if bus is not None
                },
            }
            for k, v in stor.items()
            if tag in v.tags
        ]
        for stor_et in stors_et:
            if energy_type not in stor_et["energy_type"]:
                unit_name = stor_et["name"]
                exception_list.append(
                    NetworkValidatorException(
                        f"Energy type for storages of the tag: {tag} "
                        f"for <{unit_name}> do not match energy type in Generation Fraction"
                    )
                )

    @staticmethod
    def _validate_tag_energy_types(
        min_max: dict[str, dict[tuple[str, str], float]],
        network: Network,
        exception_list: list[NetworkValidatorException],
    ) -> None:
        """Check tags energy types"""
        for energy_type, tags_gen_frac in min_max.items():
            et = {v.name: v.energy_type for k, v in network.buses.items()}
            for tags in tags_gen_frac.keys():
                for tag in tags:
                    NetworkGenerationFraction._gen_et_validation(
                        et, network, tag, energy_type, exception_list
                    )
                    NetworkGenerationFraction._stor_et_validation(
                        et, network, tag, energy_type, exception_list
                    )

    @staticmethod
    def validate(
        network: Network, exception_list: list[NetworkValidatorException]
    ) -> None:
        energy_types = network.energy_types
        min_gen_fr = network.constants.min_generation_fraction
        max_gen_fr = network.constants.max_generation_fraction
        NetworkGenerationFraction._validate_min_max_gen_frac(
            min_gen_fr, "Min", energy_types, exception_list
        )
        NetworkGenerationFraction._validate_min_max_gen_frac(
            max_gen_fr, "Max", energy_types, exception_list
        )

        for min_max_gen_fr in (min_gen_fr, max_gen_fr):
            NetworkGenerationFraction._validate_proper_subtags(
                min_max_gen_fr, network, exception_list
            )
            NetworkGenerationFraction._validate_tag_energy_types(
                min_max_gen_fr, network, exception_list
            )
