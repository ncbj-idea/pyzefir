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

from pyzefir.model.network import Network
from pyzefir.model.network_elements import AggregatedConsumer, DemandProfile, Generator
from tests.unit.optimization.gurobi.constants import N_HOURS, N_YEARS
from tests.unit.optimization.gurobi.names import EE, HEAT
from tests.unit.optimization.gurobi.test_model.test_generator.utils import (
    minimal_unit_cap,
)
from tests.unit.optimization.gurobi.test_model.utils import (
    create_default_opf_config,
    run_opt_engine,
)


@pytest.mark.parametrize(
    (
        "build_time",
        "life_time",
        "year_sample",
        "hour_sample",
        "cap_plus_indicator",
        "yearly_heat_usage",
    ),
    [
        (1, 3, np.arange(N_YEARS), np.arange(50), np.array([0, 0, 1, 0, 0]), 1e3),
        (1, 1, np.array([0, 1, 2]), np.arange(20), np.array([1, 1, 0]), 1e3),
        (0, 1, np.arange(1), np.arange(1), np.array([0]), 1e4),
        (
            0,
            2,
            np.arange(N_YEARS),
            np.array([0, 4, 5, 10, 15]),
            np.array([0, 0, 1, 0, 1]),
            1.0,
        ),
        (0, 3, np.array([0, 1]), np.array([0, 1, 43, 1000]), np.array([0, 0]), 1e-2),
        (2, 3, np.arange(N_YEARS), np.arange(100), np.array([0, 1, 0, 0, 0]), 1e6),
    ],
)
def test_capacity_evolution_with_constant_demand(
    build_time: int,
    life_time: int,
    year_sample: np.ndarray,
    hour_sample: np.ndarray,
    yearly_heat_usage: float,
    cap_plus_indicator: np.ndarray,
    network: Network,
    biomass_heat_plant: Generator,
    aggr: AggregatedConsumer,
) -> None:
    """
    Additional assumptions:
        * heat_plant base_capacity set to the minimum feasible value,
        * constant demand (same hourly demand in each year),
        * heat_plant capex is constant in years

    Conditions to check:
        * each year heat_plant capacity is constant and equals to base capacity
        * heat_plant cap_plus[y] = 0 for y != k * life_time - build_time, where k is a positive integer
        * heat_plant cap_plus[y] = base_capacity for y == k * life_time - build_time, where k is a positive integer
        * heat_plant cap_minus = 0 for all y
        * heat_plant base_cap_minus = 0 for all y
    """

    biomass_heat_plant_type = network.generator_types[
        biomass_heat_plant.energy_source_type
    ]
    biomass_heat_plant_type.capex = pd.Series(
        np.ones(network.constants.n_years) * biomass_heat_plant_type.capex[0]
    )

    biomass_heat_plant_type.build_time = build_time
    biomass_heat_plant_type.life_time = life_time

    aggr.yearly_energy_usage[HEAT] = pd.Series(
        np.ones(network.constants.n_years) * yearly_heat_usage
    )

    biomass_heat_plant.unit_base_cap = minimal_unit_cap(
        demand=network.demand_profiles[aggr.demand_profile],
        yearly_energy_usage=aggr.yearly_energy_usage,
        energy_type=HEAT,
        efficiency=biomass_heat_plant_type.efficiency[HEAT],
        hour_sample=hour_sample,
        year_sample=year_sample,
    )

    opt_config = create_default_opf_config(hour_sample, year_sample)
    engine = run_opt_engine(network, opt_config)

    heat_plant_cap = engine.results.generators_results.cap[
        biomass_heat_plant.name
    ].values.reshape(-1)
    heat_plant_cap_plus = engine.results.generators_results.cap_plus[
        biomass_heat_plant.name
    ].values.reshape(-1)
    heat_plant_cap_minus = engine.results.generators_results.cap_minus[
        biomass_heat_plant.name
    ].values
    heat_plant_cap_base_minus = engine.results.generators_results.cap_base_minus[
        biomass_heat_plant.name
    ].values

    assert np.allclose(heat_plant_cap, biomass_heat_plant.unit_base_cap)
    assert np.allclose(
        heat_plant_cap_plus, cap_plus_indicator * biomass_heat_plant.unit_base_cap
    )
    assert np.allclose(heat_plant_cap_minus, 0)
    assert np.allclose(heat_plant_cap_base_minus, 0)


@pytest.mark.parametrize(
    (
        "build_time",
        "life_time",
        "year_sample",
        "hour_sample",
        "yearly_heat_usage",
        "expected_results",
    ),
    [
        (
            1,
            1,
            np.arange(3),
            np.arange(50),
            pd.Series(np.linspace(1e3, 1e3 * 0.8, 3)),
            {
                "cap": np.array([0.4, 0.36, 0.32]),
                "cap_plus": np.array([0.36, 0.32, 0]),
                "cap_minus": np.array([0, 0, 0]),
                "base_cap_minus": np.array([0, 0, 0]),
            },
        ),
        (
            1,
            2,
            np.arange(5),
            np.arange(100),
            pd.Series(np.linspace(1e2, 1e2 * 1.4, 5)),
            {
                "cap": np.array([0.02, 0.022, 0.024, 0.026, 0.028]),
                "cap_plus": np.array([0.002, 0.022, 0.004, 0.024, 0]),
                "cap_minus": np.array([0, 0, 0, 0, 0]),
                "base_cap_minus": np.array([0, 0, 0, 0, 0]),
            },
        ),
        (
            0,
            1,
            np.arange(5),
            np.arange(100),
            pd.Series(np.linspace(1e3, 1e3 * 0.6, 5)),
            {
                "cap": np.array([0.2, 0.18, 0.16, 0.14, 0.12]),
                "cap_plus": np.array([0, 0.18, 0.16, 0.14, 0.12]),
                "cap_minus": np.array([0, 0, 0, 0, 0]),
                "base_cap_minus": np.array([0, 0, 0, 0, 0]),
            },
        ),
        (
            0,
            2,
            np.arange(5),
            np.arange(100),
            pd.Series(np.linspace(1e3, 1e3 * 0.6, 5)),
            {
                "cap": np.array([0.2, 0.18, 0.16, 0.14, 0.12]),
                "cap_plus": np.array([0, 0, 0.16, 0, 0.12]),
                "cap_minus": np.array([0, 0, 0, 0.02, 0]),
                "base_cap_minus": np.array([0, 0.02, 0, 0, 0]),
            },
        ),
    ],
)
def test_capacity_evolution_with_variable_demand(
    build_time: int,
    life_time: int,
    year_sample: np.ndarray,
    hour_sample: np.ndarray,
    yearly_heat_usage: pd.Series,
    expected_results: dict[str, np.ndarray],
    network: Network,
    biomass_heat_plant: Generator,
    aggr: AggregatedConsumer,
    demand_profile: DemandProfile,
) -> None:
    """
    Additional assumptions
        * set demand profile to constant value == 1 / (100 * #hour_sample)
        * set minimal feasible value for biomass heat plant
    Conditions to check:
        * compare biomass heat plant capacity evolution with expected results from the test
    """

    biomass_heat_plant_type = network.generator_types[
        biomass_heat_plant.energy_source_type
    ]
    biomass_heat_plant_type.build_time = build_time
    biomass_heat_plant_type.life_time = life_time

    aggr.yearly_energy_usage[HEAT] = yearly_heat_usage
    aggr.yearly_energy_usage[EE] = yearly_heat_usage
    aggr.n_consumers = pd.Series([1] * len(yearly_heat_usage))
    demand_profile.normalized_profile[HEAT] = pd.Series(
        np.ones(N_HOURS) / 100 / hour_sample.shape[0]
    )

    biomass_heat_plant.unit_base_cap = minimal_unit_cap(
        demand=network.demand_profiles[aggr.demand_profile],
        yearly_energy_usage=aggr.yearly_energy_usage,
        energy_type=HEAT,
        efficiency=biomass_heat_plant_type.efficiency[HEAT],
        hour_sample=hour_sample,
        year_sample=year_sample,
    )

    opt_config = create_default_opf_config(hour_sample, year_sample)
    engine = run_opt_engine(network, opt_config)

    heat_plant_cap = engine.results.generators_results.cap[
        biomass_heat_plant.name
    ].values.reshape(-1)
    heat_plant_cap_plus = engine.results.generators_results.cap_plus[
        biomass_heat_plant.name
    ].values.reshape(-1)
    heat_plant_cap_minus = engine.results.generators_results.cap_minus[
        biomass_heat_plant.name
    ].values
    heat_plant_cap_base_minus = engine.results.generators_results.cap_base_minus[
        biomass_heat_plant.name
    ].values.reshape(-1)

    assert np.allclose(heat_plant_cap, expected_results["cap"])
    assert np.allclose(heat_plant_cap_plus, expected_results["cap_plus"])
    assert np.allclose(heat_plant_cap_minus.sum(axis=0), expected_results["cap_minus"])
    assert np.allclose(heat_plant_cap_base_minus, expected_results["base_cap_minus"])
