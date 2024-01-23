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
from tests.unit.optimization.gurobi.constants import N_HOURS, N_YEARS
from tests.unit.optimization.gurobi.names import EE, GRID, HEAT
from tests.unit.optimization.gurobi.test_model.utils import (
    create_default_opf_config,
    run_opt_engine,
    set_network_elements_parameters,
)


@pytest.mark.parametrize(
    (
        "opt_config_params",
        "generator_params",
        "generator_type_params",
        "demand_params",
        "aggr_params",
        "expected_results",
    ),
    [
        (
            {"hour_sample": np.arange(100), "year_sample": np.arange(5)},
            {
                "heat_pump_gen": {"unit_base_cap": 1e-4},
                f"pp_coal_{GRID}": {"unit_base_cap": 2 * 1e-4},
            },
            {
                "heat_pump": {
                    "conversion_rate": {
                        EE: pd.Series(np.ones(N_HOURS)),
                    },
                    "efficiency": {HEAT: 1.0, EE: 0.0},
                },
                "pp_coal": {"efficiency": {HEAT: 0.0, EE: 1.0}},
            },
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series(np.ones(N_HOURS) / 1e4),
                        EE: pd.Series(np.ones(N_HOURS) / 1e4),
                    }
                }
            },
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.ones(N_YEARS)),
                        EE: pd.Series(np.ones(N_YEARS)),
                    }
                }
            },
            {
                "heat_pump_gen": np.ones((100, 5)) * 1e-4,
                f"pp_coal_{GRID}": np.ones((100, 5)) * 2 * 1e-4,
            },
        ),
        (
            {"hour_sample": np.arange(5), "year_sample": np.arange(2)},
            {
                "heat_pump_gen": {"unit_base_cap": 12.0},
                f"pp_coal_{GRID}": {"unit_base_cap": 14.5},
            },
            {
                "heat_pump": {
                    "conversion_rate": {
                        EE: pd.Series([10, 10, 10, 5, 3.3333] + [0] * (N_HOURS - 5)),
                    },
                    "efficiency": {HEAT: 0.5, EE: 0.0},
                },
                "pp_coal": {"efficiency": {HEAT: 0.0, EE: 0.8}},
            },
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series([1, 1, 2, 2, 3] + [0] * (N_HOURS - 5)),
                        EE: pd.Series([2, 2, 2, 5, 1] + [0] * (N_HOURS - 5)),
                    }
                }
            },
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.arange(1, 6)),
                        EE: pd.Series(np.arange(1, 6)),
                    }
                }
            },
            {
                "heat_pump_gen": np.array([[2, 2, 4, 4, 6], [4, 4, 8, 8, 12]]).T,
                f"pp_coal_{GRID}": np.array(
                    [[2.75, 2.75, 3.0, 7.25, 3.5], [5.5, 5.5, 6.0, 14.5, 7]]
                ).T,
            },
        ),
        (
            {"hour_sample": np.arange(5), "year_sample": np.arange(2)},
            {
                "heat_pump_gen": {"unit_base_cap": 12.0},
                f"pp_coal_{GRID}": {"unit_base_cap": 12.5},
            },
            {
                "heat_pump": {
                    "conversion_rate": {
                        EE: pd.Series(np.ones(N_HOURS) * 100000000),
                    },
                    "efficiency": {HEAT: 0.5, EE: 0.0},
                },
                "pp_coal": {"efficiency": {HEAT: 0.0, EE: 0.8}},
            },
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series([1, 1, 2, 2, 3] + [0] * (N_HOURS - 5)),
                        EE: pd.Series([2, 2, 2, 5, 1] + [0] * (N_HOURS - 5)),
                    }
                }
            },
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.ones(N_YEARS)),
                        EE: pd.Series([0.5, 3.0, 1, 1, 1]),
                    }
                }
            },
            {
                "heat_pump_gen": np.array([[2, 2, 4, 4, 6], [2, 2, 4, 4, 6]]).T,
                f"pp_coal_{GRID}": np.array(
                    [[1.25, 1.25, 1.25, 3.125, 0.625], [7.5, 7.5, 7.5, 18.75, 3.75]]
                ).T,
            },
        ),
    ],
    ids=(
        "constant_yearly_energy_usage_no_energy_losses",
        "variable_yearly_energy_usage_nonzero_energy_losses",
        "zero_heat_pump_conversion_rate",
    ),
)
def test_generation_values(
    opt_config_params: dict[str, np.ndarray],
    generator_params: dict[str, dict],
    generator_type_params: dict[str, dict],
    demand_params: dict[str, dict],
    aggr_params: dict[str, dict],
    expected_results: dict[str, np.ndarray],
    network: Network,
) -> None:
    """
    Test, if for given network parametrization, generation of heat pump and coal power plant are correct.
    """
    set_network_elements_parameters(network.generators, generator_params)
    set_network_elements_parameters(network.generator_types, generator_type_params)
    set_network_elements_parameters(network.demand_profiles, demand_params)
    set_network_elements_parameters(network.aggregated_consumers, aggr_params)

    opt_config = create_default_opf_config(**opt_config_params)
    engine = run_opt_engine(network, opt_config)
    results = engine.results

    for generator_name in expected_results:
        gen = results.generators_results.gen[generator_name]
        assert np.allclose(gen, expected_results[generator_name])
