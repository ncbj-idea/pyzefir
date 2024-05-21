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
from tests.unit.optimization.linopy.names import EE, GRID, HEAT
from tests.unit.optimization.linopy.test_model.utils import (
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
            {"hour_sample": np.arange(3), "year_sample": np.arange(5)},
            {
                "heat_pump_gen": {"unit_base_cap": 5.5},
                f"pp_coal_{GRID}": {"unit_base_cap": 7.8},
            },
            {
                "heat_pump": {
                    "conversion_rate": {
                        EE: pd.Series([10, 5, 1.25] + [0] * (N_HOURS - 3)),
                    },
                    "efficiency": {
                        HEAT: pd.Series([0.8, 0.8, 0.8]),
                        EE: pd.Series([0.0, 0.0, 0.0]),
                    },
                    "build_time": 0,
                },
                "pp_coal": {
                    "efficiency": {
                        HEAT: pd.Series([0.0, 0.0, 0.0]),
                        EE: pd.Series([0.5, 0.5, 0.5]),
                    },
                    "build_time": 1,
                },
            },
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series([2, 4.4, 3.9]),
                        EE: pd.Series(np.ones(N_HOURS) / 1e4),
                    }
                }
            },
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series([1, 13, 14, 15, 16]),
                        EE: pd.Series(np.zeros(N_YEARS)),
                    }
                }
            },
            {
                "heat_pump_gen": np.array([1, 13, 14, 15, 16]) * 5.5,
                f"pp_coal_{GRID}": np.array([1, 13, 14, 15, 16]) * 7.8,
            },
        ),
        (
            {"hour_sample": np.arange(3), "year_sample": np.arange(5)},
            {
                "heat_pump_gen": {"unit_base_cap": 0.0},
                f"pp_coal_{GRID}": {"unit_base_cap": 2 * 1e-4},
            },
            {
                "heat_pump": {
                    "conversion_rate": {
                        EE: pd.Series([10, 5, 1.25] + [0] * (N_HOURS - 3)),
                    },
                    "efficiency": {
                        HEAT: pd.Series([0.8, 0.8, 0.8]),
                        EE: pd.Series([0.0, 0.0, 0.0]),
                    },
                    "build_time": 0,
                },
                "pp_coal": {
                    "efficiency": {
                        HEAT: pd.Series([0.0, 0.0, 0.0]),
                        EE: pd.Series([0.5, 0.5, 0.5]),
                    },
                    "build_time": 1,
                },
            },
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series([2, 4.4, 3.9]),
                        EE: pd.Series(np.ones(N_HOURS) / 1e4),
                    }
                }
            },
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.zeros(N_YEARS)),
                        EE: pd.Series(np.ones(N_YEARS)),
                    }
                }
            },
            {"heat_pump_gen": np.zeros(5), f"pp_coal_{GRID}": np.ones(5) * 2 * 1e-4},
        ),
    ],
    ids=(
        "non_zero_heat_and_ee_yearly_energy_usage",
        "zero_heat_nonzero_ee_yearly_energy_usage",
    ),
)
def test_capacity(
    opt_config_params: dict[str, np.ndarray],
    generator_params: dict[str, dict],
    generator_type_params: dict[str, dict],
    demand_params: dict[str, dict],
    aggr_params: dict[str, dict],
    expected_results: dict[str, np.ndarray],
    network: Network,
) -> None:
    set_network_elements_parameters(network.generators, generator_params)
    set_network_elements_parameters(network.generator_types, generator_type_params)
    set_network_elements_parameters(network.demand_profiles, demand_params)
    set_network_elements_parameters(network.aggregated_consumers, aggr_params)

    opt_config = create_default_opf_config(**opt_config_params)
    engine = run_opt_engine(network, opt_config)
    results = engine.results

    for generator_name in expected_results:
        assert np.allclose(
            results.generators_results.cap[generator_name].values.reshape(-1),
            expected_results[generator_name],
        )
