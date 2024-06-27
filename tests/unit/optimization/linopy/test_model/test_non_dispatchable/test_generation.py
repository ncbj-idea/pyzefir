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
from tests.unit.optimization.linopy.constants import N_HOURS, N_YEARS
from tests.unit.optimization.linopy.names import EE, GRID, HEAT, HS
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opt_config,
    run_opt_engine,
    set_network_elements_parameters,
)


@pytest.mark.parametrize(
    (
        "config_params",
        "capacity_factor_params",
        "generator_params",
        "generator_type_params",
        "demand_params",
        "aggr_params",
        "expected_results",
    ),
    [
        pytest.param(
            {"hour_sample": np.arange(6), "year_sample": np.arange(1)},
            {
                "sun": {
                    "profile": pd.Series(
                        np.concatenate(
                            ([0.2, 0.3, 0.8, 0.5, 0.4, 0.1], np.zeros(N_HOURS - 6))
                        )
                    )
                }
            },
            {
                "local_pv": {"unit_base_cap": 1.0},
                "global_solar": {"unit_base_cap": 2.0},
            },
            {
                "pv": {"efficiency": {EE: pd.Series([1.0] * 6)}},
                "solar": {"efficiency": {HEAT: pd.Series([1.0] * 6)}},
                "pp_coal": {"efficiency": {EE: pd.Series([1.0] * 6)}},
                "heat_plant_biomass": {"efficiency": {HEAT: pd.Series([1.0] * 6)}},
            },
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series(np.ones(N_HOURS) / N_HOURS),
                        EE: pd.Series(np.ones(N_HOURS) / N_HOURS),
                    }
                }
            },
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.ones(N_YEARS) * N_HOURS),
                        EE: pd.Series(np.ones(N_YEARS) * N_HOURS),
                    }
                }
            },
            {
                f"pp_coal_{GRID}": {
                    "gen": np.array([0.8, 0.7, 0.2, 0.5, 0.6, 0.9]).reshape((-1, 1)),
                    "gen_ee": np.array([0.8, 0.7, 0.2, 0.5, 0.6, 0.9]).reshape((-1, 1)),
                    "gen_heat": np.zeros(6).reshape((-1, 1)),
                    "dump": np.zeros(6).reshape((-1, 1)),
                    "dump_ee": np.zeros(6).reshape((-1, 1)),
                    "dump_heat": np.zeros(6).reshape((-1, 1)),
                },
                "local_pv": {
                    "gen": np.array([0.2, 0.3, 0.8, 0.5, 0.4, 0.1]).reshape((-1, 1)),
                    "gen_ee": np.array([0.2, 0.3, 0.8, 0.5, 0.4, 0.1]).reshape((-1, 1)),
                    "gen_heat": np.zeros(6).reshape((-1, 1)),
                    "dump": np.zeros(6).reshape((-1, 1)),
                    "dump_ee": np.zeros(6).reshape((-1, 1)),
                    "dump_heat": np.zeros(6).reshape((-1, 1)),
                },
                "global_solar": {
                    "gen": np.array([0.4, 0.6, 1.6, 1.0, 0.8, 0.2]).reshape((-1, 1)),
                    "gen_heat": np.array([0.4, 0.6, 1.0, 1.0, 0.8, 0.2]).reshape(
                        (-1, 1)
                    ),
                    "gen_ee": np.zeros(6).reshape((-1, 1)),
                    "dump": np.array([0, 0, 0.6, 0, 0, 0]).reshape((-1, 1)),
                    "dump_heat": np.array([0, 0, 0.6, 0, 0, 0]).reshape((-1, 1)),
                    "dump_ee": np.zeros(6).reshape((-1, 1)),
                },
                f"biomass_heat_plant_{HS}": {
                    "gen": np.array([0.6, 0.4, 0.0, 0.0, 0.2, 0.8]).reshape((-1, 1)),
                    "gen_heat": np.array([0.6, 0.4, 0.0, 0.0, 0.2, 0.8]).reshape(
                        (-1, 1)
                    ),
                    "gen_ee": np.zeros(6).reshape((-1, 1)),
                    "dump": np.zeros(6).reshape((-1, 1)),
                    "dump_heat": np.zeros(6).reshape((-1, 1)),
                    "dump_ee": np.zeros(6).reshape((-1, 1)),
                },
            },
            id="equal efficiency, constant demand, variable non-negative capacity factor",
        ),
        pytest.param(
            {"hour_sample": np.arange(6), "year_sample": np.arange(1)},
            {
                "sun": {
                    "profile": pd.Series(
                        np.concatenate(
                            ([0, 0, 0.1, 0.3, 0.2, 0], np.zeros(N_HOURS - 6))
                        )
                    )
                }
            },
            {"local_pv": {"unit_base_cap": 50}, "global_solar": {"unit_base_cap": 30}},
            {
                "pv": {"efficiency": {EE: pd.Series([0.2] * 6)}},
                "solar": {"efficiency": {HEAT: pd.Series([0.5] * 6)}},
                "pp_coal": {"efficiency": {EE: pd.Series([1.0] * 6)}},
                "heat_plant_biomass": {"efficiency": {HEAT: pd.Series([1.0] * 6)}},
            },
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series(
                            np.concatenate(([3, 1, 0, 5, 7, 5], np.ones(N_HOURS - 6)))
                            / (N_HOURS + 15)
                        ),
                        EE: pd.Series(
                            np.concatenate(([2, 10, 4, 5, 1, 3], np.ones(N_HOURS - 6)))
                            / (N_HOURS + 19)
                        ),
                    }
                }
            },
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.ones(N_YEARS) * (N_HOURS + 15)),
                        EE: pd.Series(np.ones(N_YEARS) * (N_HOURS + 19)),
                    }
                }
            },
            {
                f"pp_coal_{GRID}": {
                    "gen": np.array([2, 10, 3, 2, 0, 3]).reshape((-1, 1)),
                    "gen_ee": np.array([2, 10, 3, 2, 0, 3]).reshape((-1, 1)),
                    "gen_heat": np.zeros(6).reshape((-1, 1)),
                    "dump": np.zeros(6).reshape((-1, 1)),
                    "dump_ee": np.zeros(6).reshape((-1, 1)),
                    "dump_heat": np.zeros(6).reshape((-1, 1)),
                },
                "local_pv": {
                    "gen": np.array([0, 0, 5, 15, 10, 0]).reshape((-1, 1)),
                    "gen_ee": np.array([0, 0, 1, 3, 1, 0]).reshape((-1, 1)),
                    "gen_heat": np.zeros(6).reshape((-1, 1)),
                    "dump": np.array([0, 0, 0, 0, 5, 0]).reshape((-1, 1)),
                    "dump_ee": np.array([0, 0, 0, 0, 1, 0]).reshape((-1, 1)),
                    "dump_heat": np.zeros(6).reshape((-1, 1)),
                },
                "global_solar": {
                    "gen": np.array([0, 0, 3, 9, 6, 0]).reshape((-1, 1)),
                    "gen_heat": np.array([0, 0, 0, 4.5, 3, 0]).reshape((-1, 1)),
                    "gen_ee": np.zeros(6).reshape((-1, 1)),
                    "dump": np.array([0, 0, 3, 0, 0, 0]).reshape((-1, 1)),
                    "dump_heat": np.array([0, 0, 1.5, 0, 0, 0]).reshape((-1, 1)),
                    "dump_ee": np.zeros(6).reshape((-1, 1)),
                },
                f"biomass_heat_plant_{HS}": {
                    "gen": np.array([3, 1, 0.0, 0.5, 4, 5]).reshape((-1, 1)),
                    "gen_heat": np.array([3, 1, 0.0, 0.5, 4, 5]).reshape((-1, 1)),
                    "gen_ee": np.zeros(6).reshape((-1, 1)),
                    "dump": np.zeros(6).reshape((-1, 1)),
                    "dump_heat": np.zeros(6).reshape((-1, 1)),
                    "dump_ee": np.zeros(6).reshape((-1, 1)),
                },
            },
            id="variable demand, different efficiencies, variable capacity factor with zeros",
        ),
    ],
)
def test_generation_and_dump_energy(
    config_params: dict[str, np.ndarray],
    capacity_factor_params: dict,
    generator_params: dict,
    generator_type_params: dict,
    demand_params: dict,
    aggr_params: dict,
    expected_results: dict,
    network: Network,
) -> None:
    """Test if generation and dump energy is correct."""
    set_network_elements_parameters(network.demand_profiles, demand_params)
    set_network_elements_parameters(network.capacity_factors, capacity_factor_params)
    set_network_elements_parameters(network.generators, generator_params)
    set_network_elements_parameters(network.generator_types, generator_type_params)
    set_network_elements_parameters(network.aggregated_consumers, aggr_params)

    opt_config = create_default_opt_config(**config_params)
    network.fuels["coal"].cost = network.fuels["coal"].cost / opt_config.hourly_scale
    network.fuels["biomass"].cost = (
        network.fuels["biomass"].cost / opt_config.hourly_scale
    )
    engine = run_opt_engine(network, opt_config)

    for name in expected_results:
        assert np.allclose(
            engine.results.generators_results.gen[name], expected_results[name]["gen"]
        )
        assert np.allclose(
            engine.results.generators_results.gen_et[name][EE],
            expected_results[name]["gen_ee"],
        )
        assert np.allclose(
            engine.results.generators_results.gen_et[name][HEAT],
            expected_results[name]["gen_heat"],
        )

        assert np.allclose(
            engine.results.generators_results.dump[name], expected_results[name]["dump"]
        )
        assert np.allclose(
            engine.results.generators_results.dump_et[name][EE],
            expected_results[name]["dump_ee"],
        )
        assert np.allclose(
            engine.results.generators_results.dump_et[name][HEAT],
            expected_results[name]["dump_heat"],
        )
