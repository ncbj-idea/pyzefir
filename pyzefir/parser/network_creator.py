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

from typing import Any

import numpy as np
import pandas as pd

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network import Network
from pyzefir.model.network_elements import (
    AggregatedConsumer,
    Bus,
    CapacityFactor,
    DemandChunk,
    DemandProfile,
    Fuel,
    Generator,
    GeneratorType,
    Line,
    LocalBalancingStack,
    Storage,
    StorageType,
    TransmissionFee,
)
from pyzefir.model.network_elements.dsr import DSR
from pyzefir.model.network_elements.emission_fee import EmissionFee
from pyzefir.model.utils import NetworkConstants
from pyzefir.parser.elements_parsers.aggregated_consumer_parser import (
    AggregatedConsumerParser,
)
from pyzefir.parser.elements_parsers.bus_parser import BusParser
from pyzefir.parser.elements_parsers.capacity_factor_parser import CapacityFactorParser
from pyzefir.parser.elements_parsers.demand_chunk_parser import DemandChunkParser
from pyzefir.parser.elements_parsers.demand_profile_parser import DemandProfileParser
from pyzefir.parser.elements_parsers.dsr_parser import DSRParser
from pyzefir.parser.elements_parsers.emission_fee_parser import EmissionFeeParser
from pyzefir.parser.elements_parsers.energy_source_type_parser import (
    EnergySourceTypeParser,
)
from pyzefir.parser.elements_parsers.energy_source_unit_parser import (
    EnergySourceUnitParser,
)
from pyzefir.parser.elements_parsers.fuel_parser import FuelParser
from pyzefir.parser.elements_parsers.line_parser import LineParser
from pyzefir.parser.elements_parsers.local_balancing_stack_parser import (
    LocalBalancingStackParser,
)
from pyzefir.parser.elements_parsers.transmission_fee_parser import (
    TransmissionFeeParser,
)
from pyzefir.utils.path_manager import DataCategories, DataSubCategories


class NetworkCreator:
    @staticmethod
    def create(
        df_dict: dict[str, dict[str, pd.DataFrame]],
        config_dict: dict[str, Any] | None = None,
    ) -> Network:
        network_constants = NetworkCreator._create_network_constants(
            df_dict, config_dict
        )
        fuels = NetworkCreator._create_fuels(df_dict)
        capacity_factors = NetworkCreator._create_capacity_factors(df_dict)
        generator_types, storage_types = NetworkCreator._create_energy_source_types(
            df_dict, network_constants
        )
        energy_types = NetworkCreator._create_energy_types(df_dict)
        emission_types = NetworkCreator._create_emission_types(df_dict)
        buses = NetworkCreator._create_buses(df_dict)
        generators, storages = NetworkCreator._create_energy_source_units(
            df_dict, network_constants
        )
        lines = NetworkCreator._create_lines(df_dict)
        local_balancing_stacks = NetworkCreator._create_local_balancing_stacks(df_dict)
        aggregated_consumers = NetworkCreator._create_aggregated_consumers(df_dict)
        demand_profiles = NetworkCreator._create_demand_profiles(df_dict)
        transmission_fees = NetworkCreator._create_transmission_fees(df_dict)
        emission_fees = NetworkCreator._create_emission_fees(df_dict)
        demand_chunks = NetworkCreator._create_demand_chunks(df_dict)
        dsr = NetworkCreator._create_dsr(df_dict)

        return NetworkCreator._create_network(
            network_constants,
            energy_types,
            emission_types,
            fuels,
            capacity_factors,
            buses,
            generators,
            storages,
            lines,
            local_balancing_stacks,
            aggregated_consumers,
            generator_types,
            storage_types,
            demand_profiles,
            transmission_fees,
            emission_fees,
            demand_chunks,
            dsr,
        )

    @staticmethod
    def _create_network(
        network_constants: NetworkConstants,
        energy_types: list[str],
        emission_types: list[str],
        fuels: tuple[Fuel, ...],
        capacity_factors: tuple[CapacityFactor, ...],
        buses: tuple[Bus, ...],
        generators: tuple[Generator, ...],
        storages: tuple[Storage, ...],
        lines: tuple[Line, ...],
        local_balancing_stacks: tuple[LocalBalancingStack, ...],
        aggregated_consumers: tuple[AggregatedConsumer, ...],
        generator_types: tuple[GeneratorType, ...],
        storage_types: tuple[StorageType, ...],
        demand_profiles: tuple[DemandProfile, ...],
        transmission_fees: tuple[TransmissionFee, ...],
        emission_fees: tuple[EmissionFee, ...],
        demand_chunks: tuple[DemandChunk, ...],
        dsr: tuple[DSR, ...],
    ) -> Network:
        network = Network(
            network_constants=network_constants,
            energy_types=energy_types,
            emission_types=emission_types,
        )
        exception_list: list[NetworkValidatorException] = []
        objects_and_methods_list = [
            (emission_fees, network.add_emission_fee),
            (dsr, network.add_dsr),
            (buses, network.add_bus),
            (fuels, network.add_fuel),
            (capacity_factors, network.add_capacity_factor),
            (generator_types, network.add_generator_type),
            (storage_types, network.add_storage_type),
            (generators, network.add_generator),
            (storages, network.add_storage),
            (transmission_fees, network.add_transmission_fee),
            (lines, network.add_line),
            (local_balancing_stacks, network.add_local_balancing_stack),
            (demand_profiles, network.add_demand_profile),
            (aggregated_consumers, network.add_aggregated_consumer),
            (demand_chunks, network.add_demand_chunk),
        ]
        for elements, add_method in objects_and_methods_list:
            for element in elements:
                try:
                    add_method(element)
                except NetworkValidatorException as error:
                    exception_list.append(error)

        if exception_list:
            raise NetworkValidatorExceptionGroup(
                "While creating object network following errors occurred: ",
                exception_list,
            )
        return network

    @staticmethod
    def _create_network_constants(
        df_dict: dict[str, dict[str, pd.DataFrame]],
        config_dict: dict[str, Any] | None = None,
    ) -> NetworkConstants:
        constants_df = df_dict[DataCategories.SCENARIO][DataSubCategories.CONSTANTS]
        constants_dict = constants_df.pivot_table(columns="constants_name").to_dict(
            "index"
        )["constants_value"]
        rel_emission_lim_df = df_dict[DataCategories.SCENARIO][
            DataSubCategories.RELATIVE_EMISSION_LIMITS
        ].set_index("year_idx")
        rel_emission_lim_dict = {
            col_name: rel_emission_lim_df[col_name].reindex(
                [y for y in range(0, constants_dict["N_YEARS"])]
            )
            for col_name in rel_emission_lim_df.columns
        }
        constants_dict[
            DataSubCategories.RELATIVE_EMISSION_LIMITS
        ] = rel_emission_lim_dict
        constants_dict["base_total_emission"] = (
            df_dict[DataCategories.STRUCTURE][DataSubCategories.EMISSION_TYPES]
            .pivot_table(columns="name", dropna=False)
            .to_dict("index")["base_total_emission"]
        )
        generation_fractions_df = df_dict[DataCategories.SCENARIO][
            DataSubCategories.GENERATION_FRACTION
        ]
        min_generation_fr, max_generation_fr = dict(), dict()
        for energy_type in generation_fractions_df["energy_type"].values:
            per_en_type_df = generation_fractions_df[
                generation_fractions_df["energy_type"] == energy_type
            ]
            tags = (per_en_type_df["tag"].iloc[0], per_en_type_df["subtag"].iloc[0])
            min_generation_fr.update(
                {energy_type: {tags: per_en_type_df["min_generation_fraction"].iloc[0]}}
            )
            max_generation_fr.update(
                {energy_type: {tags: per_en_type_df["max_generation_fraction"].iloc[0]}}
            )
        power_reserves = (
            df_dict[DataCategories.STRUCTURE][DataSubCategories.POWER_RESERVE]
            .pivot_table(
                index="energy_type",
                columns="tag_name",
                values="power_reserve_value",
                aggfunc="first",
            )
            .replace({np.nan: None})
            .to_dict(orient="index")
        )
        constants_dict["power_reserves"] = {
            key: {
                sub_key: float(sub_value)
                for sub_key, sub_value in value.items()
                if sub_value is not None
            }
            for key, value in power_reserves.items()
        }
        constants_dict = {k.lower(): v for k, v in constants_dict.items()}
        config_dict = config_dict if config_dict else dict()
        constants_dict["min_generation_fraction"] = min_generation_fr
        constants_dict["max_generation_fraction"] = max_generation_fr
        return NetworkConstants(**constants_dict | config_dict)

    @staticmethod
    def _create_demand_profiles(
        df_dict: dict[str, dict[str, pd.DataFrame]]
    ) -> tuple[DemandProfile, ...]:
        demand_profiles = DemandProfileParser(
            df_dict[DataCategories.DEMAND],
        ).create()
        return demand_profiles

    @staticmethod
    def _create_buses(df_dict: dict[str, dict[str, pd.DataFrame]]) -> tuple[Bus, ...]:
        buses = BusParser(
            df_dict[DataCategories.STRUCTURE][DataSubCategories.BUSES],
        ).create()
        return buses

    @staticmethod
    def _create_transmission_fees(
        df_dict: dict[str, dict[str, pd.DataFrame]]
    ) -> tuple[TransmissionFee, ...]:
        transmission_fees = TransmissionFeeParser(
            df_dict[DataCategories.STRUCTURE][DataSubCategories.TRANSMISSION_FEES],
        ).create()
        return transmission_fees

    @staticmethod
    def _create_emission_fees(
        df_dict: dict[str, dict[str, pd.DataFrame]]
    ) -> tuple[EmissionFee, ...]:
        emission_fees = EmissionFeeParser(
            df_dict[DataCategories.STRUCTURE][
                DataSubCategories.EMISSION_FEES_EMISSION_TYPES
            ],
            df_dict[DataCategories.SCENARIO][DataSubCategories.EMISSION_FEES],
        ).create()

        return emission_fees

    @staticmethod
    def _create_lines(df_dict: dict[str, dict[str, pd.DataFrame]]) -> tuple[Line, ...]:
        lines = LineParser(
            df_dict[DataCategories.STRUCTURE][DataSubCategories.LINES]
        ).create()
        return lines

    @staticmethod
    def _create_energy_source_units(
        df_dict: dict[str, dict[str, pd.DataFrame]],
        network_constants: NetworkConstants,
    ) -> tuple[tuple[Generator, ...], tuple[Storage, ...]]:
        generators, storages = EnergySourceUnitParser(
            df_generators=df_dict[DataCategories.STRUCTURE][
                DataSubCategories.GENERATORS
            ],
            df_storages=df_dict[DataCategories.STRUCTURE][DataSubCategories.STORAGES],
            df_element_energy_evolution=df_dict[DataCategories.SCENARIO][
                DataSubCategories.ELEMENT_ENERGY_EVOLUTION_LIMITS
            ],
            df_technology_bus=df_dict[DataCategories.STRUCTURE][
                DataSubCategories.TECHNOLOGY_BUS
            ],
            df_technology=df_dict[DataCategories.INITIAL_STATE][
                DataSubCategories.TECHNOLOGY
            ],
            df_tech_stack_bus=df_dict[DataCategories.STRUCTURE][
                DataSubCategories.TECHNOLOGYSTACK_BUSES
            ],
            df_tech_stack_aggregate=df_dict[DataCategories.STRUCTURE][
                DataSubCategories.TECHNOLOGYSTACK_AGGREGATE
            ],
            df_tech_stack=df_dict[DataCategories.INITIAL_STATE][
                DataSubCategories.TECHNOLOGYSTACK
            ],
            df_aggregates=df_dict[DataCategories.STRUCTURE][
                DataSubCategories.AGGREGATES
            ],
            n_years=network_constants.n_years,
            df_generator_emission_fee=df_dict[DataCategories.STRUCTURE][
                DataSubCategories.GENERATOR_EMISSION_FEES
            ],
            n_consumers=df_dict[DataCategories.SCENARIO][DataSubCategories.N_CONSUMERS],
        ).create()
        return generators, storages

    @staticmethod
    def _create_emission_types(
        df_dict: dict[str, dict[str, pd.DataFrame]]
    ) -> list[str]:
        emission_df = df_dict[DataCategories.STRUCTURE][
            DataSubCategories.EMISSION_TYPES
        ]
        return list(emission_df["name"])

    @staticmethod
    def _create_energy_types(df_dict: dict[str, dict[str, pd.DataFrame]]) -> list[str]:
        energy_df = df_dict[DataCategories.STRUCTURE][DataSubCategories.ENERGY_TYPES]
        return list(energy_df["name"])

    @staticmethod
    def _create_local_balancing_stacks(
        df_dict: dict[str, dict[str, pd.DataFrame]],
    ) -> tuple[LocalBalancingStack, ...]:
        stacks = LocalBalancingStackParser(
            df_dict[DataCategories.STRUCTURE][
                DataSubCategories.TECHNOLOGYSTACKS_BUSES_OUT
            ],
            df_dict[DataCategories.STRUCTURE][DataSubCategories.BUSES],
            df_dict[DataCategories.STRUCTURE][DataSubCategories.TECHNOLOGYSTACK_BUSES],
        ).create()

        return stacks

    @staticmethod
    def _create_aggregated_consumers(
        df_dict: dict[str, dict[str, pd.DataFrame]]
    ) -> tuple[AggregatedConsumer, ...]:
        aggregated_consumers = AggregatedConsumerParser(
            df_dict[DataCategories.STRUCTURE][DataSubCategories.AGGREGATES],
            df_dict[DataCategories.STRUCTURE][
                DataSubCategories.TECHNOLOGYSTACK_AGGREGATE
            ],
            df_dict[DataCategories.INITIAL_STATE][DataSubCategories.TECHNOLOGYSTACK],
            df_dict[DataCategories.SCENARIO][DataSubCategories.YEARLY_ENERGY_USAGE],
            df_dict[DataCategories.SCENARIO][DataSubCategories.FRACTIONS],
            df_dict[DataCategories.SCENARIO][DataSubCategories.CONSTANTS]
            .query("constants_name == 'N_YEARS'")["constants_value"]
            .squeeze(),
            df_dict[DataCategories.SCENARIO][DataSubCategories.N_CONSUMERS],
        ).create()

        return aggregated_consumers

    @staticmethod
    def _create_fuels(df_dict: dict[str, dict[str, pd.DataFrame]]) -> tuple[Fuel, ...]:
        fuels = FuelParser(
            df_dict[DataCategories.FUELS][DataSubCategories.EMISSION_PER_UNIT],
            df_dict[DataCategories.FUELS][DataSubCategories.ENERGY_PER_UNIT],
            df_dict[DataCategories.SCENARIO][DataSubCategories.FUEL_PRICES],
            df_dict[DataCategories.SCENARIO][DataSubCategories.FUEL_AVAILABILITY],
        ).create()

        return fuels

    @staticmethod
    def _create_capacity_factors(
        df_dict: dict[str, dict[str, pd.DataFrame]]
    ) -> tuple[CapacityFactor, ...]:
        capacity_factors = CapacityFactorParser(
            df_dict[DataCategories.CAPACITY_FACTORS][DataSubCategories.PROFILES]
        ).create()

        return capacity_factors

    @staticmethod
    def _create_energy_source_types(
        df_dict: dict[str, dict[str, pd.DataFrame]],
        network_constants: NetworkConstants,
    ) -> tuple[tuple[GeneratorType, ...], tuple[StorageType, ...]]:
        generator_types, storage_types = EnergySourceTypeParser(
            cost_parameters_df=df_dict[DataCategories.SCENARIO][
                DataSubCategories.COST_PARAMETERS
            ],
            storage_type_df=df_dict[DataCategories.STORAGE][
                DataSubCategories.PARAMETERS
            ],
            energy_mix_evolution_limits_df=df_dict[DataCategories.SCENARIO][
                DataSubCategories.ENERGY_SOURCE_EVOLUTION_LIMITS
            ],
            conversion_rate=df_dict[DataCategories.CONVERSION_RATE],
            generators_efficiency=df_dict[DataCategories.GENERATOR][
                DataSubCategories.EFFICIENCY
            ],
            generators_emission_reduction=df_dict[DataCategories.GENERATOR][
                DataSubCategories.EMISSION_REDUCTION
            ],
            generators_energy_type=df_dict[DataCategories.GENERATOR][
                DataSubCategories.GENERATOR_TYPE_ENERGY_TYPE
            ],
            generators_fuel_type=df_dict[DataCategories.GENERATOR][
                DataSubCategories.GENERATOR_TYPE_ENERGY_CARRIER
            ],
            generators_type=df_dict[DataCategories.GENERATOR][
                DataSubCategories.GENERATOR_TYPES
            ],
            generators_power_utilization=df_dict[DataCategories.GENERATOR][
                DataSubCategories.POWER_UTILIZATION
            ],
            n_years=network_constants.n_years,
            n_hours=network_constants.n_hours,
            curtailment_cost=df_dict[DataCategories.SCENARIO][
                DataSubCategories.CURTAILMENT_COST
            ],
        ).create()
        return generator_types, storage_types

    @staticmethod
    def _create_demand_chunks(
        df_dict: dict[str, dict[str, pd.DataFrame]]
    ) -> tuple[DemandChunk, ...]:
        demand_chunks = DemandChunkParser(
            df_dict[DataCategories.DEMAND_CHUNKS]
        ).create()

        return demand_chunks

    @classmethod
    def _create_dsr(
        cls, df_dict: dict[str, dict[str, pd.DataFrame]]
    ) -> tuple[DSR, ...]:
        dsr = DSRParser(
            df_dict[DataCategories.STRUCTURE][DataSubCategories.DSR]
        ).create()

        return dsr
