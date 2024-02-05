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
from dataclasses import dataclass, fields
from pathlib import Path

logger = logging.getLogger(__name__)


class DataCategoriesException(Exception):
    pass


class CsvPathManagerException(Exception):
    pass


@dataclass(frozen=True)
class DataCategories:
    INITIAL_STATE: str = "initial_state"
    STRUCTURE: str = "structure"
    CAPACITY_FACTORS: str = "capacity_factors"
    FUELS: str = "fuels"
    GENERATOR: str = "generator_types"
    STORAGE: str = "storage_types"
    DEMAND: str = "demand_types"
    SCENARIO: str = "scenarios"
    DEMAND_CHUNKS: str = "demand_chunks"
    CONVERSION_RATE: str = "conversion_rate"

    @classmethod
    def check_directory_name(cls, value: str) -> None:
        if not any(getattr(cls, f.name) == value for f in fields(cls)):
            logger.warning(f"Incorrect {cls.__name__} field: {value}")
            raise DataCategoriesException(
                f"{cls.__name__} does not contain a field {value}"
            )

    @staticmethod
    def get_main_categories() -> list[str]:
        return [
            DataCategories.INITIAL_STATE,
            DataCategories.STRUCTURE,
            DataCategories.CAPACITY_FACTORS,
            DataCategories.FUELS,
            DataCategories.GENERATOR,
            DataCategories.STORAGE,
            DataCategories.DEMAND,
            DataCategories.SCENARIO,
            DataCategories.CONVERSION_RATE,
            DataCategories.DEMAND_CHUNKS,
        ]

    @staticmethod
    def get_dynamic_categories() -> list[str]:
        return [
            DataCategories.DEMAND,
            DataCategories.CONVERSION_RATE,
            DataCategories.DEMAND_CHUNKS,
        ]


@dataclass(frozen=True)
class DataSubCategories:
    EMISSION_PER_UNIT: str = "Emission_Per_Unit"
    ENERGY_PER_UNIT: str = "Energy_Per_Unit"
    PROFILES: str = "Profiles"
    GENERATOR_TYPES: str = "Generator_Types"
    EFFICIENCY: str = "Efficiency"
    EMISSION_REDUCTION: str = "Emission_Reduction"
    GENERATOR_TYPE_ENERGY_CARRIER: str = "Generator_Type_Energy_Carrier"
    GENERATOR_TYPE_ENERGY_TYPE: str = "Generator_Type_Energy_Type"
    PARAMETERS: str = "Parameters"
    TECHNOLOGY: str = "Technology"
    TECHNOLOGYSTACK: str = "TechnologyStack"
    ENERGY_TYPES: str = "Energy_Types"
    EMISSION_TYPES: str = "Emission_Types"
    AGGREGATES: str = "Aggregates"
    LINES: str = "Lines"
    BUSES: str = "Buses"
    GENERATORS: str = "Generators"
    STORAGES: str = "Storages"
    TECHNOLOGYSTACKS_BUSES_OUT: str = "TechnologyStack_Buses_out"
    TECHNOLOGY_BUS: str = "Technology_Bus"
    TECHNOLOGYSTACK_BUSES: str = "TechnologyStack_Buses"
    TECHNOLOGYSTACK_AGGREGATE: str = "TechnologyStack_Aggregate"
    ENERGY_SOURCE_EVOLUTION_LIMITS: str = "Energy_Source_Evolution_Limits"
    ELEMENT_ENERGY_EVOLUTION_LIMITS: str = "Element_Energy_Evolution_Limits"
    COST_PARAMETERS: str = "Cost_Parameters"
    FUEL_AVAILABILITY: str = "Fuel_Availability"
    RELATIVE_EMISSION_LIMITS: str = "Relative_Emission_Limits"
    FUEL_PRICES: str = "Fuel_Prices"
    CONSTANTS: str = "Constants"
    YEARLY_ENERGY_USAGE: str = "Yearly_Demand"
    TRANSMISSION_FEES: str = "Transmission_Fees"
    FRACTIONS: str = "Fractions"
    N_CONSUMERS: str = "N_Consumers"
    EMISSION_FEES_EMISSION_TYPES: str = "Emission_Fees_Emission_Types"
    GENERATOR_EMISSION_FEES: str = "Generator_Emission_Fees"
    EMISSION_FEES: str = "Emission_Fees"
    DEMAND_CHUNKS: str = "Demand_Chunks"
    GENERATION_FRACTION: str = "Generation_Fraction"
    CURTAILMENT_COST: str = "Curtailment_Cost"
    DSR: str = "DSR"
    POWER_RESERVE: str = "Power_Reserve"
    POWER_UTILIZATION: str = "Power_Utilization"

    @classmethod
    def check_directory_name(cls, value: str) -> None:
        if not any(getattr(cls, f.name) == value for f in fields(cls)):
            logger.warning(f"Incorrect {cls.__name__} field: {value}")
            raise DataCategoriesException(
                f"{cls.__name__} does not contain a field {value} "
            )


def get_datasets_from_categories(data_category: str) -> list[str]:
    datasets_in_categories = {
        DataCategories.FUELS: [
            DataSubCategories.EMISSION_PER_UNIT,
            DataSubCategories.ENERGY_PER_UNIT,
        ],
        DataCategories.CAPACITY_FACTORS: [
            DataSubCategories.PROFILES,
        ],
        DataCategories.GENERATOR: [
            DataSubCategories.GENERATOR_TYPES,
            DataSubCategories.EFFICIENCY,
            DataSubCategories.EMISSION_REDUCTION,
            DataSubCategories.GENERATOR_TYPE_ENERGY_CARRIER,
            DataSubCategories.GENERATOR_TYPE_ENERGY_TYPE,
            DataSubCategories.POWER_UTILIZATION,
        ],
        DataCategories.STORAGE: [DataSubCategories.PARAMETERS],
        DataCategories.INITIAL_STATE: [
            DataSubCategories.TECHNOLOGY,
            DataSubCategories.TECHNOLOGYSTACK,
        ],
        DataCategories.STRUCTURE: [
            DataSubCategories.ENERGY_TYPES,
            DataSubCategories.EMISSION_TYPES,
            DataSubCategories.AGGREGATES,
            DataSubCategories.LINES,
            DataSubCategories.BUSES,
            DataSubCategories.GENERATORS,
            DataSubCategories.STORAGES,
            DataSubCategories.TECHNOLOGYSTACKS_BUSES_OUT,
            DataSubCategories.TECHNOLOGY_BUS,
            DataSubCategories.TECHNOLOGYSTACK_BUSES,
            DataSubCategories.TECHNOLOGYSTACK_AGGREGATE,
            DataSubCategories.TRANSMISSION_FEES,
            DataSubCategories.EMISSION_FEES_EMISSION_TYPES,
            DataSubCategories.GENERATOR_EMISSION_FEES,
            DataSubCategories.DSR,
            DataSubCategories.POWER_RESERVE,
        ],
        DataCategories.SCENARIO: [
            DataSubCategories.ELEMENT_ENERGY_EVOLUTION_LIMITS,
            DataSubCategories.ENERGY_SOURCE_EVOLUTION_LIMITS,
            DataSubCategories.COST_PARAMETERS,
            DataSubCategories.FUEL_AVAILABILITY,
            DataSubCategories.RELATIVE_EMISSION_LIMITS,
            DataSubCategories.FUEL_PRICES,
            DataSubCategories.CONSTANTS,
            DataSubCategories.YEARLY_ENERGY_USAGE,
            DataSubCategories.FRACTIONS,
            DataSubCategories.N_CONSUMERS,
            DataSubCategories.EMISSION_FEES,
            DataSubCategories.GENERATION_FRACTION,
            DataSubCategories.CURTAILMENT_COST,
        ],
        DataCategories.DEMAND_CHUNKS: [DataSubCategories.DEMAND_CHUNKS],
    }

    try:
        return datasets_in_categories[data_category]
    except KeyError:
        logger.warning(f"{data_category=} not in datasets_in_categories keys")
        raise


class CsvPathManager:
    def __init__(self, dir_path: Path, scenario_name: str | None = None) -> None:
        self._dir_path = dir_path
        self._scenario_name = scenario_name

    def __repr__(self) -> str:
        return f"CsvPathManager({self._dir_path=})"

    @property
    def dir_path(self) -> Path:
        logger.debug(f"Csv root dir path is {self._dir_path}")
        return self._dir_path

    def get_path(self, data_category: str, dataset_name: str | None = None) -> Path:
        DataCategories.check_directory_name(data_category)
        if dataset_name:
            if data_category == DataCategories.SCENARIO and self._scenario_name:
                target_path = self._dir_path.joinpath(
                    data_category,
                    f"{self._scenario_name}/{self._get_file_name_from_dict(data_category, dataset_name)}",
                )
            else:
                target_path = self._dir_path.joinpath(
                    data_category,
                    self._get_file_name_from_dict(data_category, dataset_name),
                )
            logger.debug(f"File {dataset_name} is at the path: {target_path}")
        else:
            target_path = self._dir_path.joinpath(data_category)
            logger.debug(f"Given folder is at the path: {target_path}")
        return target_path

    def concatenate_path_for_dynamic_dataset_name(
        self, category: str, dataset_name: str
    ) -> Path:
        root_path = self.get_path(data_category=category)
        return root_path.joinpath(f"{dataset_name}.csv")

    @staticmethod
    def _get_file_name_from_dict(data_category: str, dataset_name: str) -> str:
        try:
            DataSubCategories.check_directory_name(dataset_name)
            return f"{dataset_name}.csv"
        except DataCategoriesException as e:
            logger.warning(f"Exception was raised: {e}")
            raise CsvPathManagerException(
                f"File {dataset_name} in category {data_category} is not part of given structure"
            )


class XlsxPathManager(CsvPathManager):
    def __init__(
        self, input_path: Path, output_path: Path, scenario_name: str | None = None
    ) -> None:
        super().__init__(output_path)
        self._input_path = input_path
        self._scenario_name = scenario_name

    def __repr__(self) -> str:
        return f"XlsxPathManager({self.input_path=}, {self.output_path=})"

    @property
    def input_path(self) -> Path:
        logger.debug(f"Input path is {self._input_path}")
        return self._input_path

    @property
    def output_path(self) -> Path:
        logger.debug(f"Output path is {self._dir_path}")
        return self._dir_path

    def get_input_file_path(self, data_category: str) -> Path:
        DataCategories.check_directory_name(data_category)
        target_path = self._input_path.joinpath(f"{data_category}.xlsx")

        logger.debug(f"Path for file {data_category}: {target_path}")
        return target_path
