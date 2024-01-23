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
from pyzefir.model.network_elements import Line
from tests.unit.optimization.gurobi.constants import N_HOURS, N_YEARS
from tests.unit.optimization.gurobi.names import EE, GRID, HEAT, HS
from tests.unit.optimization.gurobi.test_model.utils import (
    create_default_opf_config,
    run_opt_engine,
    set_network_elements_parameters,
)


@pytest.mark.parametrize(
    (
        "config_params",
        "line_params",
        "demand_params",
        "aggr_params",
        "expected_results",
    ),
    [
        pytest.param(
            {"hour_sample": np.arange(4), "year_sample": np.arange(1)},
            {
                f"{GRID}->local_ee_bus": {"transmission_loss": 0.1},
                f"{HS}->local_heat_bus": {"transmission_loss": 0.2},
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
                        EE: pd.Series(np.ones(N_YEARS) * N_HOURS * 2),
                    }
                }
            },
            {
                f"{HS}->local_heat_bus": np.ones((4, 1)) * 1.0 / 0.8,
                f"{GRID}->local_ee_bus": np.ones((4, 1)) * 2.0 / 0.9,
            },
            id="constant profile, equal demand, non-zero losses",
        ),
        pytest.param(
            {"hour_sample": np.arange(4), "year_sample": np.arange(1)},
            {
                f"{GRID}->local_ee_bus": {"transmission_loss": 0.9},
                f"{HS}->local_heat_bus": {"transmission_loss": 0.1},
            },
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series(
                            np.concatenate(([2, 2, 5, 3], np.ones(N_HOURS - 4)))
                            / (12 + N_HOURS - 4)
                        ),
                        EE: pd.Series(
                            np.concatenate(([3, 1, 4, 1], np.ones(N_HOURS - 4)))
                            / (9 + N_HOURS - 4)
                        ),
                    }
                }
            },
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.ones(N_YEARS) * (12 + N_HOURS - 4)),
                        EE: pd.Series(np.ones(N_YEARS) * (9 + N_HOURS - 4) * 2),
                    }
                }
            },
            {
                f"{HS}->local_heat_bus": np.array([2, 2, 5, 3]).reshape((-1, 1)) / 0.9,
                f"{GRID}->local_ee_bus": np.array([3, 1, 4, 1]).reshape((-1, 1))
                * 2
                / 0.1,
            },
            id="constant profile, equal demand, non-zero losses",
        ),
    ],
)
def test_flow_values(
    config_params: dict[str, np.ndarray],
    line_params: dict,
    demand_params: dict,
    aggr_params: dict,
    expected_results: dict[str, np.ndarray],
    grid_connection: Line,
    heating_system_connection: Line,
    network: Network,
) -> None:
    """Test if flow on lines connecting lbs with KSE and heating system is correct for different loss parameters."""
    set_network_elements_parameters(network.aggregated_consumers, aggr_params)
    set_network_elements_parameters(network.demand_profiles, demand_params)
    set_network_elements_parameters(network.lines, line_params)

    opt_config = create_default_opf_config(**config_params)
    engine = run_opt_engine(network, opt_config)

    for line_name in line_params:
        assert np.allclose(
            engine.results.lines_results.flow[line_name], expected_results[line_name]
        )
