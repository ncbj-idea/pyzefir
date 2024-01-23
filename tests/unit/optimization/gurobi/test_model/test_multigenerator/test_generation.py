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
from pyzefir.model.network_elements import Generator
from tests.unit.optimization.gurobi.constants import N_HOURS, N_YEARS
from tests.unit.optimization.gurobi.names import EE, HEAT
from tests.unit.optimization.gurobi.test_model.utils import (
    create_default_opf_config,
    run_opt_engine,
    set_network_elements_parameters,
)


@pytest.mark.parametrize(
    (
        "config_params",
        "generator_type_params",
        "demand_params",
        "aggr_params",
        "expected_results",
    ),
    [
        pytest.param(
            {"hour_sample": np.arange(5), "year_sample": np.arange(2)},
            {"chp_coal": {"efficiency": {HEAT: 0.5, EE: 0.5}}},
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
                "gen": np.ones(10).reshape((5, 2)) * 2,
                "gen_h": np.ones(10).reshape((5, 2)),
                "gen_ee": np.ones(10).reshape((5, 2)),
                "dump_h": np.zeros(10).reshape((5, 2)),
                "dump_ee": np.zeros(10).reshape((5, 2)),
            },
            id="equal demands equal efficiency (constant profile)",
        ),
        pytest.param(
            {"hour_sample": np.arange(5), "year_sample": np.arange(2)},
            {"chp_coal": {"efficiency": {HEAT: 0.2, EE: 0.7}}},
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series(np.ones(N_YEARS) / N_HOURS),
                        EE: pd.Series(np.ones(N_YEARS) / N_HOURS),
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
                "gen": np.ones(10).reshape((5, 2)) * 5,
                "gen_h": np.ones(10).reshape((5, 2)),
                "gen_ee": np.ones(10).reshape((5, 2)),
                "dump_h": np.zeros(10).reshape((5, 2)),
                "dump_ee": np.ones(10).reshape((5, 2)) * (5 * 0.7 - 1),
            },
            id="equal demands different efficiency (constant profile)",
        ),
        pytest.param(
            {"hour_sample": np.arange(5), "year_sample": np.arange(2)},
            {"chp_coal": {"efficiency": {HEAT: 0.5, EE: 0.5}}},
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series(np.ones(N_YEARS) / N_HOURS),
                        EE: pd.Series(np.ones(N_YEARS) / N_HOURS),
                    }
                }
            },
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.ones(N_YEARS) * N_HOURS),
                        EE: pd.Series(np.ones(N_YEARS) * N_HOURS * 3),
                    }
                }
            },
            {
                "gen": np.ones(10).reshape((5, 2)) * 6,
                "gen_h": np.ones(10).reshape((5, 2)),
                "gen_ee": np.ones(10).reshape((5, 2)) * 3,
                "dump_h": np.ones(10).reshape((5, 2)) * 2,
                "dump_ee": np.zeros(10).reshape((5, 2)),
            },
            id="different demands equal efficiency (constant profile)",
        ),
        pytest.param(
            {"hour_sample": np.arange(5), "year_sample": np.arange(2)},
            {"chp_coal": {"efficiency": {HEAT: 0.2, EE: 0.7}}},
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series(np.ones(N_YEARS) / N_HOURS),
                        EE: pd.Series(np.ones(N_YEARS) / N_HOURS),
                    }
                }
            },
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.ones(N_YEARS) * N_HOURS * 2),
                        EE: pd.Series(np.ones(N_YEARS) * N_HOURS),
                    }
                }
            },
            {
                "gen": np.ones(10).reshape((5, 2)) * 10,
                "gen_h": np.ones(10).reshape((5, 2)) * 2,
                "gen_ee": np.ones(10).reshape((5, 2)),
                "dump_h": np.zeros(10).reshape((5, 2)),
                "dump_ee": np.ones(10).reshape((5, 2)) * (10 * 0.7 - 1),
            },
            id="different demands different efficiency (constant profile)",
        ),
        pytest.param(
            {"hour_sample": np.arange(5), "year_sample": np.arange(2)},
            {"chp_coal": {"efficiency": {HEAT: 0.3, EE: 0.4}}},
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series(
                            np.concatenate(([1, 2, 2, 5, 3], np.ones(N_HOURS - 5)))
                            / (13 + N_HOURS - 5)
                        ),
                        EE: pd.Series(
                            np.concatenate(([3, 1, 4, 1, 1], np.ones(N_HOURS - 5)))
                            / (10 + N_HOURS - 5)
                        ),
                    }
                }
            },
            {
                "aggr": {
                    "yearly_energy_usage": {
                        HEAT: pd.Series(np.ones(N_YEARS) * (13 + N_HOURS - 5)),
                        EE: pd.Series(np.ones(N_YEARS) * (10 + N_HOURS - 5)),
                    }
                }
            },
            {
                "gen": np.array(
                    [
                        [3 / 0.4, 3 / 0.4],
                        [2 / 0.3, 2 / 0.3],
                        [4 / 0.4, 4 / 0.4],
                        [5 / 0.3, 5 / 0.3],
                        [3 / 0.3, 3 / 0.3],
                    ]
                ),
                "gen_h": np.array([[1, 1], [2, 2], [2, 2], [5, 5], [3, 3]]),
                "gen_ee": np.array([[3, 3], [1, 1], [4, 4], [1, 1], [1, 1]]),
                "dump_h": np.array(
                    [
                        [(3 / 0.4) * 0.3 - 1, (3 / 0.4) * 0.3 - 1],
                        [0, 0],
                        [(4 / 0.4) * 0.3 - 2, (4 / 0.4) * 0.3 - 2],
                        [0, 0],
                        [0, 0],
                    ]
                ),
                "dump_ee": np.array(
                    [
                        [0, 0],
                        [(2 / 0.3) * 0.4 - 1, (2 / 0.3) * 0.4 - 1],
                        [0, 0],
                        [(5 / 0.3) * 0.4 - 1, (5 / 0.3) * 0.4 - 1],
                        [(3 / 0.3) * 0.4 - 1, (3 / 0.3) * 0.4 - 1],
                    ]
                ),
            },
            id="different demands different efficiency (variable profile)",
        ),
    ],
)
def test_gen_and_de(
    config_params: dict[str, np.ndarray],
    generator_type_params: dict,
    demand_params: dict,
    aggr_params: dict,
    expected_results: dict[str, np.ndarray],
    coal_chp: Generator,
    network: Network,
) -> None:
    """Test if generation and dump energy are correct."""
    set_network_elements_parameters(network.aggregated_consumers, aggr_params)
    set_network_elements_parameters(network.demand_profiles, demand_params)
    set_network_elements_parameters(network.generator_types, generator_type_params)

    opt_config = create_default_opf_config(**config_params)
    engine = run_opt_engine(network, opt_config)

    assert np.allclose(
        engine.results.generators_results.gen[coal_chp.name], expected_results["gen"]
    )
    assert np.allclose(
        engine.results.generators_results.gen_et[coal_chp.name][HEAT],
        expected_results["gen_h"],
    )
    assert np.allclose(
        engine.results.generators_results.gen_et[coal_chp.name][EE],
        expected_results["gen_ee"],
    )
    assert np.allclose(
        engine.results.generators_results.dump_et[coal_chp.name][HEAT],
        expected_results["dump_h"],
    )
    assert np.allclose(
        engine.results.generators_results.dump_et[coal_chp.name][EE],
        expected_results["dump_ee"],
    )
