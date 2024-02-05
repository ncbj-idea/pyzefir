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
from dataclasses import dataclass
from typing import Type

from pyzefir.utils.path_manager import DataCategories, DataSubCategories

DataFramesColumnsType = Type[str | int | float | bool]
ColumnType = dict[str, DataFramesColumnsType]

logger = logging.getLogger(__name__)


class InvalidStructureException(Exception):
    pass


@dataclass
class DatasetConfig:
    columns: ColumnType
    dataset_name: str
    default_type: set[DataFramesColumnsType] | None = None


def get_dataset_config_from_categories(
    data_category: str, dataset_name: str
) -> DatasetConfig:
    dataset_validation_structure: dict[str, list[DatasetConfig] | DatasetConfig] = {
        DataCategories.FUELS: [
            DatasetConfig(
                dataset_name=DataSubCategories.EMISSION_PER_UNIT,
                columns={"name": str},
                default_type={float},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.ENERGY_PER_UNIT,
                columns={"name": str, "energy_per_unit": float},
            ),
        ],
        DataCategories.CAPACITY_FACTORS: [
            DatasetConfig(
                dataset_name=DataSubCategories.PROFILES,
                columns={"hour_idx": int},
                default_type={float},
            )
        ],
        DataCategories.GENERATOR: [
            DatasetConfig(
                dataset_name=DataSubCategories.GENERATOR_TYPES,
                columns={
                    "name": str,
                    "ramp": float,
                    "build_time": int,
                    "life_time": int,
                    "power_utilization": float,
                },
                default_type={bool},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.EFFICIENCY,
                columns={
                    "generator_type": str,
                    "energy_type": str,
                    "efficiency": float,
                },
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.EMISSION_REDUCTION,
                columns={"generator_type": str},
                default_type={float},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.GENERATOR_TYPE_ENERGY_CARRIER,
                columns={
                    "generator_type": str,
                    "fuel_name": str,
                    "capacity_factor_name": str,
                },
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.GENERATOR_TYPE_ENERGY_TYPE,
                columns={"generator_type": str, "energy_type": str},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.POWER_UTILIZATION,
                columns={"hour_idx": int},
                default_type={float},
            ),
        ],
        DataCategories.STORAGE: [
            DatasetConfig(
                dataset_name=DataSubCategories.PARAMETERS,
                columns={
                    "storage_type": str,
                    "load_efficiency": float,
                    "gen_efficiency": float,
                    "cycle_length": int,
                    "power_to_capacity": int,
                    "energy_type": str,
                    "energy_loss": float,
                    "build_time": int,
                    "life_time": int,
                    "power_utilization": float,
                },
                default_type={bool},
            )
        ],
        DataCategories.INITIAL_STATE: [
            DatasetConfig(
                dataset_name=DataSubCategories.TECHNOLOGY,
                columns={"technology": str, "base_capacity": float},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.TECHNOLOGYSTACK,
                columns={
                    "technology_stack": str,
                    "aggregate": str,
                    "base_fraction": float,
                },
            ),
        ],
        DataCategories.STRUCTURE: [
            DatasetConfig(
                dataset_name=DataSubCategories.ENERGY_TYPES, columns={"name": str}
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.EMISSION_TYPES,
                columns={"name": str, "base_total_emission": float},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.AGGREGATES,
                columns={
                    "name": str,
                    "demand_type": str,
                    "n_consumers_base": int,
                    "average_area": float,
                },
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.LINES,
                columns={
                    "name": str,
                    "energy_type": str,
                    "bus_from": str,
                    "bus_to": str,
                    "transmission_loss": float,
                    "max_capacity": float,
                    "transmission_fee": str,
                },
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.BUSES,
                columns={"name": str, "energy_type": str, "dsr_type": str},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.GENERATORS,
                columns={
                    "name": str,
                    "generator_type": str,
                    "min_device_nom_power": float,
                    "max_device_nom_power": float,
                },
                default_type={bool},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.STORAGES,
                columns={
                    "name": str,
                    "storage_type": str,
                    "min_device_nom_power": float,
                    "max_device_nom_power": float,
                },
                default_type={bool},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.TECHNOLOGYSTACKS_BUSES_OUT,
                columns={"name": str},
                default_type={str},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.TECHNOLOGY_BUS,
                columns={"technology": str, "type": str, "bus": str},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.TECHNOLOGYSTACK_BUSES,
                columns={"technology_stack": str, "bus": str},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.TECHNOLOGYSTACK_AGGREGATE,
                columns={"technology_stack": str, "aggregate": str},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.TRANSMISSION_FEES,
                columns={"hour_idx": int},
                default_type={float},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.EMISSION_FEES_EMISSION_TYPES,
                columns={"emission_type": str, "emission_fee": str},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.GENERATOR_EMISSION_FEES,
                columns={"generator": str, "emission_fee": str},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.DSR,
                columns={
                    "name": str,
                    "compensation_factor": float,
                    "balancing_period_len": int,
                    "penalization": float,
                    "relative_shift_limit": float,
                    "abs_shift_limit": float,
                },
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.POWER_RESERVE,
                columns={
                    "tag_name": str,
                    "energy_type": str,
                    "power_reserve_value": float,
                },
            ),
        ],
        DataCategories.SCENARIO: [
            DatasetConfig(
                dataset_name=DataSubCategories.ENERGY_SOURCE_EVOLUTION_LIMITS,
                columns={
                    "year_idx": int,
                    "technology_type": str,
                    "max_capacity": float,
                    "min_capacity": float,
                    "max_capacity_increase": float,
                    "min_capacity_increase": float,
                },
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.ELEMENT_ENERGY_EVOLUTION_LIMITS,
                columns={
                    "year_idx": int,
                    "technology_name": str,
                    "max_capacity": float,
                    "min_capacity": float,
                    "max_capacity_increase": float,
                    "min_capacity_increase": float,
                },
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.GENERATION_FRACTION,
                columns={
                    "tag": str,
                    "subtag": str,
                    "energy_type": str,
                    "min_generation_fraction": float,
                    "max_generation_fraction": float,
                },
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.COST_PARAMETERS,
                columns={
                    "year_idx": int,
                    "technology_type": str,
                    "CAPEX": float,
                    "OPEX": float,
                },
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.FUEL_AVAILABILITY,
                columns={"year_idx": int},
                default_type={float, int},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.RELATIVE_EMISSION_LIMITS,
                columns={"year_idx": int},
                default_type={float},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.FUEL_PRICES,
                columns={"year_idx": int},
                default_type={float},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.CONSTANTS,
                columns={"constants_name": str, "constants_value": int},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.YEARLY_ENERGY_USAGE,
                columns={
                    "aggregate": str,
                    "energy_type": str,
                    "year_idx": int,
                    "value": float,
                },
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.FRACTIONS,
                columns={
                    "technology_stack": str,
                    "aggregate": str,
                    "year": int,
                    "min_fraction": float,
                    "max_fraction": float,
                    "max_fraction_increase": float,
                    "max_fraction_decrease": float,
                },
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.N_CONSUMERS,
                columns={"year_idx": int},
                default_type={int},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.CURTAILMENT_COST,
                columns={"year_idx": int},
                default_type={float},
            ),
            DatasetConfig(
                dataset_name=DataSubCategories.EMISSION_FEES,
                columns={"year_idx": int},
                default_type={float},
            ),
        ],
        DataCategories.DEMAND: DatasetConfig(
            dataset_name=dataset_name, columns={"hour_idx": int}, default_type={float}
        ),
        DataCategories.CONVERSION_RATE: DatasetConfig(
            dataset_name=dataset_name,
            columns={
                "hour_idx": int,
            },
            default_type={float},
        ),
        DataCategories.DEMAND_CHUNKS: [
            DatasetConfig(
                dataset_name=DataSubCategories.DEMAND_CHUNKS,
                columns={
                    "name": str,
                    "tag": str,
                    "energy_type": str,
                },
                default_type={float},
            ),
            DatasetConfig(
                dataset_name=dataset_name,
                columns={
                    "period_start": int,
                    "period_end": int,
                },
                default_type={int},
            ),
        ],
    }

    if (selected_category := dataset_validation_structure.get(data_category)) is None:
        logger.warning(f"{data_category=} not in dataset_validation_structure keys")
        raise InvalidStructureException(
            f"{data_category=} not in dataset_validation_structure keys"
        )
    if isinstance(selected_category, DatasetConfig):
        return selected_category
    elif isinstance(selected_category, list):
        for data_config in selected_category:
            if data_config.dataset_name == dataset_name:
                return data_config
    logger.warning(
        f"No dataset config found for category {data_category} and dataset {dataset_name}"
    )
    raise InvalidStructureException(
        f"No dataset config found for category {data_category} and dataset {dataset_name}"
    )


def get_dataset_reference(category: str, dataset_name: str) -> str:
    return f"{category} / {dataset_name}"
