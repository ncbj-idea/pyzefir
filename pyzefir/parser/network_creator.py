import logging
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
    GenerationFraction,
    Generator,
    GeneratorType,
    Line,
    LocalBalancingStack,
    Storage,
    StorageType,
    TransmissionFee,
)
from pyzefir.model.network_elements.capacity_bound import CapacityBound
from pyzefir.model.network_elements.dsr import DSR
from pyzefir.model.network_elements.emission_fee import EmissionFee
from pyzefir.model.utils import NetworkConstants
from pyzefir.parser.elements_parsers.aggregated_consumer_parser import (
    AggregatedConsumerParser,
)
from pyzefir.parser.elements_parsers.bus_parser import BusParser
from pyzefir.parser.elements_parsers.capacity_bound_parser import CapacityBoundParser
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
from pyzefir.parser.elements_parsers.generation_fraction_parser import (
    GenerationFractionParser,
)
from pyzefir.parser.elements_parsers.line_parser import LineParser
from pyzefir.parser.elements_parsers.local_balancing_stack_parser import (
    LocalBalancingStackParser,
)
from pyzefir.parser.elements_parsers.transmission_fee_parser import (
    TransmissionFeeParser,
)
from pyzefir.utils.path_manager import DataCategories, DataSubCategories

_logger = logging.getLogger(__name__)


class NetworkCreator:
    """
    A class for creating a network from various input data frames.

    The NetworkCreator class provides methods to build a comprehensive energy network model.
    It processes input data to extract various components of the network, including constants,
    energy sources, demand profiles, and more. This class also validates the created network to ensure
    all elements are correctly integrated.
    """

    @staticmethod
    def create(
        df_dict: dict[str, dict[str, pd.DataFrame]],
        config_dict: dict[str, Any] | None = None,
    ) -> Network:
        """
        Create a network using provided data frames and configuration.

        This method orchestrates the network creation process by calling various helper methods
        to create network components from the input data. It validates the created components
        and raises exceptions if any errors occur during the process.

        Args:
            - df_dict: A dictionary containing data categorized for network creation.
            - config_dict: A dictionary containing configuration parameters (optional).

        Returns:
            - Network: A fully constructed Network object.
        """
        _logger.info("Creating network...")
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
        generation_fractions = NetworkCreator._create_generation_fraction(
            df_dict, network_constants
        )
        dsr = NetworkCreator._create_dsr(df_dict)
        capacity_bounds = NetworkCreator._create_capacity_bounds(df_dict)
        _logger.info("Network creation: Done")

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
            capacity_bounds,
            generation_fractions,
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
        capacity_bounds: tuple[CapacityBound, ...],
        generation_fractions: tuple[GenerationFraction, ...],
    ) -> Network:
        """
        Construct the network from its components.

        This method assembles various components into a complete network object and performs
        validation checks on the elements being added. If any issues are encountered, they are
        collected and raised as exceptions.

        Args:
            - network_constants (NetworkConstants): Constants required for the network.
            - energy_types (list[str]): A list of energy types used in the network.
            - emission_types (list[str]): A list of emission types used in the network.
            - fuels (tuple[Fuel, ...]): Tuple of Fuel objects.
            - capacity_factors (tuple[CapacityFactor, ...]): Tuple of CapacityFactor objects.
            - buses (tuple[Bus, ...]): Tuple of Bus objects.
            - generators (tuple[Generator, ...]): Tuple of Generator objects.
            - storages (tuple[Storage, ...]): Tuple of Storage objects.
            - lines (tuple[Line, ...]): Tuple of Line objects.
            - local_balancing_stacks (tuple[LocalBalancingStack, ...]): Tuple of LocalBalancingStack objects.
            - aggregated_consumers (tuple[AggregatedConsumer, ...]): Tuple of AggregatedConsumer objects.
            - generator_types (tuple[GeneratorType, ...]): Tuple of GeneratorType objects.
            - storage_types (tuple[StorageType, ...]): Tuple of StorageType objects.
            - demand_profiles (tuple[DemandProfile, ...]): Tuple of DemandProfile objects.
            - transmission_fees (tuple[TransmissionFee, ...]): Tuple of TransmissionFee objects.
            - emission_fees (tuple[EmissionFee, ...]): Tuple of EmissionFee objects.
            - demand_chunks (tuple[DemandChunk, ...]): Tuple of DemandChunk objects.
            - dsr (tuple[DSR, ...]): Tuple of DSR objects.
            - capacity_bounds (tuple[CapacityBound, ...]): Tuple of CapacityBound objects.
            - generation_fractions (tuple[GenerationFraction, ...]): Tuple of GenerationFraction objects.

        Returns:
            - Network: A fully constructed Network object.
        """
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
            (capacity_bounds, network.add_capacity_bound),
            (generation_fractions, network.add_generation_fraction),
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
        """
        Generate network constants from the input data.

        This method extracts and processes various constants needed for the network from the
        input data frames, including relative emission limits and power reserves. It compiles
        all constants into a dictionary and returns it as a NetworkConstants object.

        Args:
            - df_dict (dict[str, dict[str, pd.DataFrame]]): A dictionary containing data categorized
                for network creation.
            - config_dict (dict[str, Any] | None): A dictionary containing configuration parameters.

        Returns:
            - NetworkConstants: A NetworkConstants object containing the extracted constants.
        """
        config_dict = config_dict if config_dict else dict()
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
        constants_dict[DataSubCategories.RELATIVE_EMISSION_LIMITS] = (
            rel_emission_lim_dict
        )
        constants_dict["base_total_emission"] = (
            df_dict[DataCategories.STRUCTURE][DataSubCategories.EMISSION_TYPES]
            .pivot_table(columns="name", dropna=False)
            .to_dict("index")["base_total_emission"]
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
        constants_dict["ens_energy_penalization"] = (
            NetworkCreator._create_ens_energy_penalization(
                df_dict[DataCategories.SCENARIO][DataSubCategories.ENS_PENALIZATION],
                df_dict[DataCategories.STRUCTURE][DataSubCategories.ENERGY_TYPES],
                config_dict.get("ens_penalty_cost", 0.0),
            )
        )
        constants_dict = {k.lower(): v for k, v in constants_dict.items()}
        _logger.info("Create network constants: Done")
        return NetworkConstants(**constants_dict | config_dict)

    @staticmethod
    def _create_ens_energy_penalization(
        ens_penalization: pd.DataFrame,
        energy_types_df: pd.DataFrame,
        penalty_cost: float,
    ) -> dict[str, float]:
        """
        Create EnsEnergyPenalization objects from the input data.

        Args:
            - ens_energy_penalization_df (pd.DataFrame): Data frame containing energy penalization data.

        Returns:
            - list[EnsEnergyPenalization]: A list of EnsEnergyPenalization objects created from the data.
        """
        ens_energy_penalization_dict = ens_penalization.set_index("energy_type")[
            "penalization"
        ].to_dict()
        return {
            et: (
                ens_energy_penalization_dict[et]
                if et in ens_energy_penalization_dict
                and not pd.isna(ens_energy_penalization_dict[et])
                and ens_energy_penalization_dict[et] >= 0.0
                else penalty_cost
            )
            for et in energy_types_df["name"].to_list()
        }

    @staticmethod
    def _create_demand_profiles(
        df_dict: dict[str, dict[str, pd.DataFrame]]
    ) -> tuple[DemandProfile, ...]:
        """
        Create DemandProfile objects from the input data.

        Args:
            - df_dict (dict[str, dict[str, pd.DataFrame]]): A dictionary containing data categorized
                for network creation.

        Returns:
            - tuple[DemandProfile, ...]: A tuple of DemandProfile objects created from the data.
        """
        demand_profiles = DemandProfileParser(
            df_dict[DataCategories.DEMAND],
        ).create()
        _logger.info("Creating demand profiles: Done")
        return demand_profiles

    @staticmethod
    def _create_buses(df_dict: dict[str, dict[str, pd.DataFrame]]) -> tuple[Bus, ...]:
        """
        Create Bus objects from the input data.

        Args:
            - df_dict (dict[str, dict[str, pd.DataFrame]]): A dictionary containing data categorized
                for network creation.

        Returns:
            - tuple[Bus, ...]: A tuple of Bus objects created from the data.
        """
        buses = BusParser(
            df_dict[DataCategories.STRUCTURE][DataSubCategories.BUSES],
        ).create()
        _logger.info("Create buses: Done")
        return buses

    @staticmethod
    def _create_transmission_fees(
        df_dict: dict[str, dict[str, pd.DataFrame]]
    ) -> tuple[TransmissionFee, ...]:
        """
        Create TransmissionFee objects from the input data.

        Args:
            - df_dict (dict[str, dict[str, pd.DataFrame]]): A dictionary containing data categorized
                for network creation.

        Returns:
            - tuple[TransmissionFee, ...]: A tuple of TransmissionFee objects created from the data.
        """
        transmission_fees = TransmissionFeeParser(
            df_dict[DataCategories.STRUCTURE][DataSubCategories.TRANSMISSION_FEES],
        ).create()
        _logger.info("Creating transmission fees: Done")
        return transmission_fees

    @staticmethod
    def _create_emission_fees(
        df_dict: dict[str, dict[str, pd.DataFrame]]
    ) -> tuple[EmissionFee, ...]:
        """
        Create EmissionFee objects from the input data.

        Args:
            - df_dict (dict[str, dict[str, pd.DataFrame]]): A dictionary containing data categorized
                for network creation.

        Returns:
            - tuple[EmissionFee, ...]: A tuple of EmissionFee objects created from the data.
        """
        emission_fees = EmissionFeeParser(
            df_dict[DataCategories.STRUCTURE][
                DataSubCategories.EMISSION_FEES_EMISSION_TYPES
            ],
            df_dict[DataCategories.SCENARIO][DataSubCategories.EMISSION_FEES],
        ).create()

        return emission_fees

    @staticmethod
    def _create_lines(df_dict: dict[str, dict[str, pd.DataFrame]]) -> tuple[Line, ...]:
        """
        Create Line objects from the input data.

        Args:
            - df_dict (dict[str, dict[str, pd.DataFrame]]): A dictionary containing data categorized
                for network creation.

        Returns:
            - tuple[Line, ...]: A tuple of Line objects created from the data.
        """
        lines = LineParser(
            df_dict[DataCategories.STRUCTURE][DataSubCategories.LINES]
        ).create()
        _logger.info("Creating lines: Done")
        return lines

    @staticmethod
    def _create_energy_source_units(
        df_dict: dict[str, dict[str, pd.DataFrame]],
        network_constants: NetworkConstants,
    ) -> tuple[tuple[Generator, ...], tuple[Storage, ...]]:
        """
        Create energy source units from the input data.

        Args:
            - df_dict (dict[str, dict[str, pd.DataFrame]]): A dictionary containing data categorized
                for network creation.
            - network_constants (NetworkConstants): Constants required for creating energy source units.

        Returns:
            - tuple[tuple[Generator, ...], tuple[Storage, ...]]: A tuple containing two tuples:
              one for generators and one for storages.
        """
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
            df_binding=df_dict[DataCategories.STRUCTURE][
                DataSubCategories.GENERATOR_BINDING
            ],
        ).create()
        _logger.info("Creating energy source units: Done")
        return generators, storages

    @staticmethod
    def _create_emission_types(
        df_dict: dict[str, dict[str, pd.DataFrame]]
    ) -> list[str]:
        """
        Extract emission types from the input data.

        Args:
            - df_dict (dict[str, dict[str, pd.DataFrame]]): A dictionary containing data categorized
                for network creation.

        Returns:
            - list[str]: A list of emission types extracted from the data.
        """
        emission_df = df_dict[DataCategories.STRUCTURE][
            DataSubCategories.EMISSION_TYPES
        ]
        _logger.info("Creating emission types: Done")
        return list(emission_df["name"])

    @staticmethod
    def _create_energy_types(df_dict: dict[str, dict[str, pd.DataFrame]]) -> list[str]:
        """
        Extract energy types from the input data.

        Args:
            - df_dict (dict[str, dict[str, pd.DataFrame]]): A dictionary containing data categorized
                for network creation.

        Returns:
            - list[str]: A list of energy types extracted from the data.
        """
        energy_df = df_dict[DataCategories.STRUCTURE][DataSubCategories.ENERGY_TYPES]
        _logger.info("Creating energy types: Done")
        return list(energy_df["name"])

    @staticmethod
    def _create_local_balancing_stacks(
        df_dict: dict[str, dict[str, pd.DataFrame]],
    ) -> tuple[LocalBalancingStack, ...]:
        """
        Create LocalBalancingStack objects from the input data.

        Args:
            - df_dict (dict[str, dict[str, pd.DataFrame]]): A dictionary containing data categorized
                for network creation.

        Returns:
            - tuple[LocalBalancingStack, ...]: A tuple of LocalBalancingStack objects created from the data.
        """
        stacks = LocalBalancingStackParser(
            df_dict[DataCategories.STRUCTURE][
                DataSubCategories.TECHNOLOGYSTACKS_BUSES_OUT
            ],
            df_dict[DataCategories.STRUCTURE][DataSubCategories.BUSES],
            df_dict[DataCategories.STRUCTURE][DataSubCategories.TECHNOLOGYSTACK_BUSES],
        ).create()
        _logger.info("Creating local balancing stacks: Done")
        return stacks

    @staticmethod
    def _create_aggregated_consumers(
        df_dict: dict[str, dict[str, pd.DataFrame]]
    ) -> tuple[AggregatedConsumer, ...]:
        """
        Create AggregatedConsumer objects from the input data.

        Args:
            - df_dict (dict[str, dict[str, pd.DataFrame]]): A dictionary containing data categorized
                for network creation.

        Returns:
            - tuple[AggregatedConsumer, ...]: A tuple of AggregatedConsumer objects created from the data.
        """
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
        _logger.info("Creating aggregated consumers: Done")
        return aggregated_consumers

    @staticmethod
    def _create_fuels(df_dict: dict[str, dict[str, pd.DataFrame]]) -> tuple[Fuel, ...]:
        """
        Create Fuel objects from the input data.

        Args:
            - df_dict (dict[str, dict[str, pd.DataFrame]]): A dictionary containing data categorized
                for network creation.

        Returns:
            - tuple[Fuel, ...]: A tuple of Fuel objects created from the data.
        """
        fuels = FuelParser(
            df_dict[DataCategories.FUELS][DataSubCategories.EMISSION_PER_UNIT],
            df_dict[DataCategories.FUELS][DataSubCategories.ENERGY_PER_UNIT],
            df_dict[DataCategories.SCENARIO][DataSubCategories.FUEL_PRICES],
            df_dict[DataCategories.SCENARIO][DataSubCategories.FUEL_AVAILABILITY],
        ).create()
        _logger.info("Creating fuels: Done")
        return fuels

    @staticmethod
    def _create_capacity_factors(
        df_dict: dict[str, dict[str, pd.DataFrame]]
    ) -> tuple[CapacityFactor, ...]:
        """
        Create CapacityFactor objects from the input data.

        Args:
            - df_dict (dict[str, dict[str, pd.DataFrame]]): A dictionary containing data categorized
                for network creation.

        Returns:
            - tuple[CapacityFactor, ...]: A tuple of CapacityFactor objects created from the data.
        """
        capacity_factors = CapacityFactorParser(
            df_dict[DataCategories.CAPACITY_FACTORS][DataSubCategories.PROFILES]
        ).create()
        _logger.info("Creating capacity factors: Done")
        return capacity_factors

    @staticmethod
    def _create_energy_source_types(
        df_dict: dict[str, dict[str, pd.DataFrame]],
        network_constants: NetworkConstants,
    ) -> tuple[tuple[GeneratorType, ...], tuple[StorageType, ...]]:
        """
        Create energy source types from the input data.

        Args:
            - df_dict (dict[str, dict[str, pd.DataFrame]]): A dictionary containing data categorized
                for network creation.
            - network_constants (NetworkConstants): Constants required for creating energy source types.

        Returns:
            - tuple[tuple[GeneratorType, ...], tuple[StorageType, ...]]: A tuple containing two tuples:
              one for generator types and one for storage types.
        """
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
            generators_series_efficiency=df_dict[
                DataCategories.GENERATOR_TYPE_EFFICIENCY
            ],
            generation_compensation=df_dict[DataCategories.SCENARIO][
                DataSubCategories.GENERATION_COMPENSATION
            ],
            yearly_emission_reduction=df_dict[DataCategories.SCENARIO][
                DataSubCategories.YEARLY_EMISSION_REDUCTION
            ],
            generators_minimal_power_utilization=df_dict[DataCategories.GENERATOR][
                DataSubCategories.MINIMAL_POWER_UTILIZATION
            ],
            storage_calculation_settings=df_dict[DataCategories.STORAGE][
                DataSubCategories.STORAGE_CALCULATION_SETTINGS
            ],
        ).create()
        _logger.info("Creating energy source types: Done")
        return generator_types, storage_types

    @staticmethod
    def _create_demand_chunks(
        df_dict: dict[str, dict[str, pd.DataFrame]]
    ) -> tuple[DemandChunk, ...]:
        """
        Create DemandChunk objects from the input data.

        Args:
            - df_dict (dict[str, dict[str, pd.DataFrame]]): A dictionary containing data categorized
                for network creation.

        Returns:
            - tuple[DemandChunk, ...]: A tuple of DemandChunk objects created from the data.
        """
        demand_chunks = DemandChunkParser(
            df_dict[DataCategories.DEMAND_CHUNKS]
        ).create()
        _logger.info("Creating demand chunks: Done")
        return demand_chunks

    @classmethod
    def _create_dsr(
        cls, df_dict: dict[str, dict[str, pd.DataFrame]]
    ) -> tuple[DSR, ...]:
        """
        Create DSR objects from the input data.

        Args:
            - df_dict (dict[str, dict[str, pd.DataFrame]]): A dictionary containing data categorized
                for network creation.

        Returns:
            - tuple[DSR, ...]: A tuple of DSR objects created from the data.
        """
        dsr = DSRParser(
            df_dict[DataCategories.STRUCTURE][DataSubCategories.DSR]
        ).create()
        _logger.info("Creating dsr: Done")
        return dsr

    @classmethod
    def _create_capacity_bounds(
        cls, df_dict: dict[str, dict[str, pd.DataFrame]]
    ) -> tuple[CapacityBound, ...]:
        """
        Create CapacityBound objects from the input data.

        Args:
            - df_dict (dict[str, dict[str, pd.DataFrame]]): A dictionary containing data categorized
                for network creation.

        Returns:
            - tuple[CapacityBound, ...]: A tuple of CapacityBound objects created from the data.
        """
        capacity_bounds = CapacityBoundParser(
            df_dict[DataCategories.SCENARIO][DataSubCategories.CAPACITY_BOUNDS]
        ).create()
        _logger.info("Creating capacity bounds: Done")
        return capacity_bounds

    @classmethod
    def _create_generation_fraction(
        cls,
        df_dict: dict[str, dict[str, pd.DataFrame]],
        network_constants: NetworkConstants,
    ) -> tuple[GenerationFraction, ...]:
        """
        Create GenerationFraction objects from the input data.

        Args:
            - df_dict (dict[str, dict[str, pd.DataFrame]]): A dictionary containing data categorized
                for network creation.
            - network_constants (NetworkConstants): Constants required for creating generation fractions.

        Returns:
            - tuple[GenerationFraction, ...]: A tuple of GenerationFraction objects created from the data.
        """
        generation_fractions = GenerationFractionParser(
            df_dict[DataCategories.SCENARIO][DataSubCategories.GENERATION_FRACTION],
            network_constants.n_years,
        ).create()
        _logger.info("Creating generation fractions: Done")
        return generation_fractions
