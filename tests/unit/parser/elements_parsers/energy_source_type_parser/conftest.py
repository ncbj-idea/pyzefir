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

from pyzefir.parser.elements_parsers.energy_source_type_parser import (
    EnergySourceTypeParser,
)
from tests.unit.defaults import ELECTRICITY, HEATING, default_network_constants


@pytest.fixture
def storage_types_mock() -> pd.DataFrame:
    """storage_types / Parameters.csv mock"""
    return pd.DataFrame(
        columns=[
            "storage_type",
            "load_efficiency",
            "gen_efficiency",
            "cycle_length",
            "power_to_capacity",
            "energy_type",
            "energy_loss",
            "build_time",
            "life_time",
            "power_utilization",
        ],
        data=[
            ["STORAGE_TYPE_1", 0.89, 0.92, 48, 7, HEATING, 0.0, 0, 10, 0.9],
            ["STORAGE_TYPE_2", 0.86, 0.94, 24, 9, ELECTRICITY, 0.0, 0, 10, 0.9],
            ["STORAGE_TYPE_3", 0.82, 0.88, 2190, 10, HEATING, 0.0, 1, 15, 0.9],
        ],
    )


@pytest.fixture
def generator_types_mock() -> pd.DataFrame:
    """generator_types / Generator_Types.csv mock"""
    return pd.DataFrame(
        columns=["name", "build_time", "life_time", "power_utilization"],
        data=[
            ["GEN_TYPE_1", 0, 20, 0.9],
            ["GEN_TYPE_2", 1, 30, 0.9],
            ["GEN_TYPE_3", 0, 15, 0.9],
        ],
    )


@pytest.fixture
def generator_type_efficiency_mock() -> pd.DataFrame:
    """generator_types / Efficiency.csv mock"""
    return pd.DataFrame(
        columns=["generator_type", "energy_type", "efficiency"],
        data=[
            ["GEN_TYPE_1", HEATING, 0.3],
            ["GEN_TYPE_1", ELECTRICITY, 0.4],
            ["GEN_TYPE_2", ELECTRICITY, 0.9],
            ["GEN_TYPE_3", HEATING, 0.84],
        ],
    )


@pytest.fixture
def emission_reduction_mock() -> pd.DataFrame:
    """generator_types / Emission_Reduction.csv mock"""
    return pd.DataFrame(
        columns=["generator_type", "CO2", "SO2"],
        data=[
            ["GEN_TYPE_1", 0.2, 0.3],
            ["GEN_TYPE_2", 0.3, 0.1],
            ["GEN_TYPE_3", 0.4, 0.25],
        ],
    )


@pytest.fixture
def generator_type_energy_type_mock() -> pd.DataFrame:
    """generator_types / Generator_Type_Energy_Type.csv mock"""
    return pd.DataFrame(
        columns=["generator_type", "energy_type"],
        data=[
            ["GEN_TYPE_1", HEATING],
            ["GEN_TYPE_1", ELECTRICITY],
            ["GEN_TYPE_2", ELECTRICITY],
            ["GEN_TYPE_3", HEATING],
        ],
    )


@pytest.fixture
def generator_fuel_type_mock() -> pd.DataFrame:
    """generator_types / Generator_Type_Energy_Carrier.csv mock"""
    return pd.DataFrame(
        columns=["generator_type", "fuel_name", "capacity_factor_name"],
        data=[["GEN_TYPE_1", "coal", np.nan], ["GEN_TYPE_2", np.nan, "sun"]],
    )


@pytest.fixture
def cost_parameters_mock() -> pd.DataFrame:
    """scenario_folder / Cost_Parameters.csv mock"""
    return pd.DataFrame(
        columns=["year_idx", "technology_type", "CAPEX", "OPEX"],
        data=[
            [0, "GEN_TYPE_1", 100, 20],
            [1, "GEN_TYPE_1", 90, 18],
            [2, "GEN_TYPE_1", 80, 15],
            [0, "GEN_TYPE_2", 120, 30],
            [1, "GEN_TYPE_2", 100, 25],
            [2, "GEN_TYPE_2", 95, 22],
            [0, "GEN_TYPE_3", 90, 15],
            [1, "GEN_TYPE_3", 88, 14],
            [2, "GEN_TYPE_3", 85, 12],
            [0, "STORAGE_TYPE_1", 200, 10],
            [1, "STORAGE_TYPE_1", 180, 10],
            [2, "STORAGE_TYPE_1", 150, 8],
            [0, "STORAGE_TYPE_2", 160, 20],
            [1, "STORAGE_TYPE_2", 150, 17],
            [2, "STORAGE_TYPE_2", 145, 13],
            [0, "STORAGE_TYPE_3", 200, 10],
            [1, "STORAGE_TYPE_3", 150, 5],
            [2, "STORAGE_TYPE_3", 120, 3],
        ],
    )


@pytest.fixture
def energy_source_evolution_limits_mock() -> pd.DataFrame:
    """scenario_folder / Energy_Source_Evolution_Limits.csv mock"""

    return pd.DataFrame(
        columns=[
            "year_idx",
            "technology_type",
            "max_capacity",
            "min_capacity",
            "max_capacity_increase",
            "min_capacity_increase",
        ],
        data=[
            [0, "GEN_TYPE_1", np.nan, np.nan, np.nan, 3],
            [1, "GEN_TYPE_1", np.nan, np.nan, np.nan, 4],
            [2, "GEN_TYPE_1", np.nan, np.nan, np.nan, 5],
            [0, "GEN_TYPE_2", 1, 1, 2, np.nan],
            [1, "GEN_TYPE_2", 1, 0.5, 2, np.nan],
            [2, "GEN_TYPE_2", 1, 0.25, 2, np.nan],
            [0, "STORAGE_TYPE_1", np.nan, 2, np.nan, np.nan],
            [1, "STORAGE_TYPE_1", np.nan, 2, np.nan, np.nan],
            [2, "STORAGE_TYPE_1", np.nan, 2, np.nan, np.nan],
            [0, "STORAGE_TYPE_2", 1, 1, np.nan, np.nan],
            [1, "STORAGE_TYPE_2", 2, 1, np.nan, np.nan],
            [2, "STORAGE_TYPE_2", 3, 1, np.nan, np.nan],
        ],
    )


@pytest.fixture
def conversion_rate_mock() -> dict[str, pd.DataFrame]:
    return {
        "GEN_TYPE_3": pd.DataFrame(
            columns=["hour_idx", ELECTRICITY],
            data=[[hour_idx, 1.0] for hour_idx in range(8760)],
        )
    }


@pytest.fixture
def energy_source_type_parser(
    storage_types_mock: pd.DataFrame,
    generator_types_mock: pd.DataFrame,
    generator_type_efficiency_mock: pd.DataFrame,
    emission_reduction_mock: pd.DataFrame,
    generator_type_energy_type_mock: pd.DataFrame,
    generator_fuel_type_mock: pd.DataFrame,
    conversion_rate_mock: dict[str, pd.DataFrame],
    cost_parameters_mock: pd.DataFrame,
    energy_source_evolution_limits_mock: pd.DataFrame,
) -> EnergySourceTypeParser:
    return EnergySourceTypeParser(
        cost_parameters_df=cost_parameters_mock,
        storage_type_df=storage_types_mock,
        generators_type=generator_types_mock,
        energy_mix_evolution_limits_df=energy_source_evolution_limits_mock,
        conversion_rate=conversion_rate_mock,
        generators_efficiency=generator_type_efficiency_mock,
        generators_emission_reduction=emission_reduction_mock,
        generators_energy_type=generator_type_energy_type_mock,
        generators_fuel_type=generator_fuel_type_mock,
        n_years=default_network_constants.n_years,
    )
