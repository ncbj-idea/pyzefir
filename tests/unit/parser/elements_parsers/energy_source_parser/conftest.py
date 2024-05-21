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

import numpy as np
import pandas as pd
import pytest

from pyzefir.parser.elements_parsers.energy_source_unit_parser import (
    EnergySourceUnitParser,
)
from pyzefir.utils.path_manager import DataCategories, DataSubCategories
from tests.unit.defaults import default_network_constants


@pytest.fixture
def generators_mock_df() -> pd.DataFrame:
    """structure / Generators.csv mock"""
    return pd.DataFrame(
        [
            {
                "name": "GENERATOR_1",
                "generator_type": "GEN_TYPE_1",
                "min_device_nom_power": 1.0,
                "max_device_nom_power": 10.0,
            },
            {
                "name": "GENERATOR_2",
                "generator_type": "GEN_TYPE_1",
                "min_device_nom_power": 5,
                "max_device_nom_power": 15,
            },
            {
                "name": "GENERATOR_3",
                "generator_type": "GEN_TYPE_2",
                "min_device_nom_power": np.nan,
                "max_device_nom_power": np.nan,
            },
        ]
    )


@pytest.fixture
def storages_mock_df() -> pd.DataFrame:
    """structure / Storages.csv mock"""
    return pd.DataFrame(
        [
            {
                "name": "STORAGE_1",
                "storage_type": "EE_STORAGE",
                "min_device_nom_power": 4,
                "max_device_nom_power": 20,
            },
            {
                "name": "STORAGE_2",
                "storage_type": "HEAT_STORAGE",
                "min_device_nom_power": 3,
                "max_device_nom_power": 3,
            },
            {
                "name": "STORAGE_3",
                "storage_type": "EE_STORAGE",
                "min_device_nom_power": 8,
                "max_device_nom_power": 22,
            },
        ]
    )


@pytest.fixture
def technology_bus_mock_df() -> pd.DataFrame:
    """structure / Technology_Bus.csv mock"""
    return pd.DataFrame(
        [
            {"technology": "STORAGE_1", "bus": "ALA", "type": "STORAGE"},
            {"technology": "STORAGE_2", "bus": "HAS", "type": "STORAGE"},
            {"technology": "STORAGE_3", "bus": "CAT", "type": "STORAGE"},
            {"technology": "GENERATOR_1", "bus": "BLE", "type": "GENERATOR"},
            {"technology": "GENERATOR_2", "bus": "BLE", "type": "GENERATOR"},
            {"technology": "GENERATOR_2", "bus": "BLE_2", "type": "GENERATOR"},
            {"technology": "GENERATOR_3", "bus": "BLE_2", "type": "GENERATOR"},
        ]
    )


@pytest.fixture
def technology_initial_state_mock_df() -> pd.DataFrame:
    """initial_state / Technology.csv mock"""
    return pd.DataFrame(
        [
            {"technology": "STORAGE_1", "base_capacity": 100000},
            {"technology": "STORAGE_2", "base_capacity": np.nan},
            {"technology": "STORAGE_3", "base_capacity": 200000},
            {"technology": "GENERATOR_1", "base_capacity": 35000},
            {"technology": "GENERATOR_2", "base_capacity": 200000},
            {"technology": "GENERATOR_3", "base_capacity": np.nan},
        ]
    )


@pytest.fixture
def empty_technology_evolution_mock_df() -> pd.DataFrame:
    """empty scenario_dir / Element_Energy_Evolution_Limits.csv mock"""
    return pd.DataFrame(
        data=[],
        columns=[
            "year_idx",
            "technology_name",
            "max_capacity",
            "min_capacity",
            "max_capacity_increase",
            "min_capacity_increase",
        ],
    )


@pytest.fixture
def technology_evolution_mock_df() -> pd.DataFrame:
    """scenario_dir / Element_Energy_Evolution_Limits.csv mock"""
    return pd.DataFrame(
        columns=[
            "year_idx",
            "technology_name",
            "max_capacity",
            "min_capacity",
            "max_capacity_increase",
            "min_capacity_increase",
        ],
        data=[
            [0, "GENERATOR_1", np.nan, np.nan, np.nan, 3],
            [1, "GENERATOR_1", np.nan, np.nan, np.nan, 4],
            [2, "GENERATOR_1", np.nan, np.nan, np.nan, 5],
            [3, "GENERATOR_1", np.nan, np.nan, np.nan, 2],
            [0, "GENERATOR_2", 1, 1, 2, np.nan],
            [1, "GENERATOR_2", 1, 1, 2, np.nan],
            [2, "GENERATOR_2", 1, 1, 2, np.nan],
            [3, "GENERATOR_2", 1, 1, 2, np.nan],
            [0, "STORAGE_1", np.nan, 2, np.nan, np.nan],
            [1, "STORAGE_1", np.nan, 2, np.nan, np.nan],
            [2, "STORAGE_1", np.nan, 2, np.nan, np.nan],
            [3, "STORAGE_1", np.nan, 2, np.nan, np.nan],
            [0, "STORAGE_2", 1, 1, np.nan, np.nan],
            [1, "STORAGE_2", 1, 1, np.nan, np.nan],
            [2, "STORAGE_2", 1, 1, np.nan, np.nan],
            [3, "STORAGE_2", 1, 1, np.nan, np.nan],
        ],
    )


@pytest.fixture
def technology_stack_initial_state_mock_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "technology_stack": "ALA",
                "aggregate": "MULTI_FAMILY",
                "base_fraction": 0.8,
            },
            {
                "technology_stack": "HAS",
                "aggregate": "MULTI_FAMILY",
                "base_fraction": 0.1,
            },
            {
                "technology_stack": "CAT",
                "aggregate": "SINGLE_FAMILY",
                "base_fraction": 0.5,
            },
            {
                "technology_stack": "BLE",
                "aggregate": "SINGLE_FAMILY",
                "base_fraction": 0.7,
            },
            {
                "technology_stack": "BLE_2",
                "aggregate": "MULTI_FAMILY",
                "base_fraction": 0.3,
            },
        ]
    )


@pytest.fixture
def technology_stack_bus_mock_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"technology_stack": "ALA", "bus": "ALA"},
            {"technology_stack": "HAS", "bus": "HAS"},
            {"technology_stack": "CAT", "bus": "CAT"},
            {"technology_stack": "BLE", "bus": "BLE"},
            {"technology_stack": "BLE", "bus": "BLE"},
            {"technology_stack": "BLE", "bus": "BLE"},
        ]
    )


@pytest.fixture
def technology_stack_aggregate_mock() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"technology_stack": "ALA", "aggregate": "MULTI_FAMILY"},
            {"technology_stack": "HAS", "aggregate": "MULTI_FAMILY"},
            {"technology_stack": "CAT", "aggregate": "SINGLE_FAMILY"},
            {"technology_stack": "BLE", "aggregate": "SINGLE_FAMILY"},
            {"technology_stack": "BLE_2", "aggregate": "MULTI_FAMILY"},
        ]
    )


@pytest.fixture
def aggregates_mock_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": "MULTI_FAMILY",
                "demand_type": "MULTI_FAMILY",
                "n_consumers_base": 30000,
            },
            {
                "name": "SINGLE_FAMILY",
                "demand_type": "MULTI_FAMILY",
                "n_consumers_base": 50000,
            },
        ]
    )


@pytest.fixture
def generator_emission_fee_mock_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"generator": "GENERATOR_1", "emission_fee": "EF_1"},
            {"generator": "GENERATOR_1", "emission_fee": "EF_2"},
            {"generator": "GENERATOR_1", "emission_fee": "EF_3"},
            {"generator": "GENERATOR_2", "emission_fee": "EF_1"},
        ]
    )


@pytest.fixture
def generator_binding_mock_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"generator": "GENERATOR_1", "binding_name": "mock_binding_name"},
            {"generator": "GENERATOR_2", "binding_name": "mock_binding_name"},
        ]
    )


@pytest.fixture
def df_dict(
    generators_mock_df: pd.DataFrame,
    storages_mock_df: pd.DataFrame,
    technology_bus_mock_df: pd.DataFrame,
    technology_initial_state_mock_df: pd.DataFrame,
    empty_technology_evolution_mock_df: pd.DataFrame,
    technology_stack_initial_state_mock_df: pd.DataFrame,
    aggregates_mock_df: pd.DataFrame,
    technology_stack_bus_mock_df: pd.DataFrame,
    technology_stack_aggregate_mock: pd.DataFrame,
    generator_emission_fee_mock_df: pd.DataFrame,
    generator_binding_mock_df: pd.DataFrame,
) -> dict[str, dict[str, pd.DataFrame]]:
    return {
        DataCategories.INITIAL_STATE: {
            DataSubCategories.TECHNOLOGY: technology_initial_state_mock_df,
            DataSubCategories.TECHNOLOGYSTACK: technology_stack_initial_state_mock_df,
        },
        DataCategories.STRUCTURE: {
            DataSubCategories.AGGREGATES: aggregates_mock_df,
            DataSubCategories.GENERATORS: generators_mock_df,
            DataSubCategories.STORAGES: storages_mock_df,
            DataSubCategories.TECHNOLOGY_BUS: technology_bus_mock_df,
            DataSubCategories.TECHNOLOGYSTACK_BUSES: technology_stack_bus_mock_df,
            DataSubCategories.TECHNOLOGYSTACK_AGGREGATE: technology_stack_aggregate_mock,
            DataSubCategories.GENERATOR_EMISSION_FEES: generator_emission_fee_mock_df,
            DataSubCategories.GENERATOR_BINDING: generator_binding_mock_df,
        },
        DataCategories.SCENARIO: {
            DataSubCategories.ELEMENT_ENERGY_EVOLUTION_LIMITS: empty_technology_evolution_mock_df,
            DataSubCategories.N_CONSUMERS: pd.DataFrame(columns=["year_idx"]),
        },
    }


@pytest.fixture
def energy_source_unit_parser(
    df_dict: dict[str, dict[str, pd.DataFrame]]
) -> EnergySourceUnitParser:
    """mock EnergySourceUnitParser object"""
    return EnergySourceUnitParser(
        df_generators=df_dict[DataCategories.STRUCTURE][DataSubCategories.GENERATORS],
        df_storages=df_dict[DataCategories.STRUCTURE][DataSubCategories.STORAGES],
        df_technology_bus=df_dict[DataCategories.STRUCTURE][
            DataSubCategories.TECHNOLOGY_BUS
        ],
        df_technology=df_dict[DataCategories.INITIAL_STATE][
            DataSubCategories.TECHNOLOGY
        ],
        df_element_energy_evolution=df_dict[DataCategories.SCENARIO][
            DataSubCategories.ELEMENT_ENERGY_EVOLUTION_LIMITS
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
        df_aggregates=df_dict[DataCategories.STRUCTURE][DataSubCategories.AGGREGATES],
        n_years=default_network_constants.n_years,
        df_generator_emission_fee=df_dict[DataCategories.STRUCTURE][
            DataSubCategories.GENERATOR_EMISSION_FEES
        ],
        n_consumers=df_dict[DataCategories.SCENARIO][DataSubCategories.N_CONSUMERS],
        df_binding=df_dict[DataCategories.STRUCTURE][
            DataSubCategories.GENERATOR_BINDING
        ],
    )
