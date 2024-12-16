from typing import Any

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.network import Network
from pyzefir.optimization.linopy.model import OptimizationError
from tests.unit.optimization.linopy.constants import N_YEARS
from tests.unit.optimization.linopy.names import EE, HEAT
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opt_config,
    run_opt_engine,
    set_network_elements_parameters,
)


@pytest.mark.parametrize(
    "aggr_yearly_energy_usage, generators_params, excepted_capacity",
    [
        pytest.param(
            {
                EE: pd.Series([0] * N_YEARS),
                HEAT: pd.Series([0] * N_YEARS),
            },
            {
                "pp_coal_grid": {
                    "unit_base_cap": 1000,
                    "unit_min_capacity": pd.Series([np.nan] + [1100] * (N_YEARS - 1)),
                    "unit_max_capacity": pd.Series([np.nan] + [1100] * (N_YEARS - 1)),
                    "unit_min_capacity_increase": pd.Series([np.nan] * N_YEARS),
                    "unit_max_capacity_increase": pd.Series([np.nan] * N_YEARS),
                }
            },
            {
                "pp_coal_grid": [1000, 1100, 1100, 1100, 1100],
            },
            id="capacity_should_be_1100_for_each_year_except_first_one",
        ),
        pytest.param(
            {
                EE: pd.Series([0] * N_YEARS),
                HEAT: pd.Series([0] * N_YEARS),
            },
            {
                "pp_coal_grid": {
                    "unit_base_cap": 1000,
                    "unit_min_capacity": pd.Series([np.nan] * N_YEARS),
                    "unit_max_capacity": pd.Series([np.nan] * N_YEARS),
                    "unit_min_capacity_increase": pd.Series(
                        [np.nan] + [-100] * (N_YEARS - 1)
                    ),
                    "unit_max_capacity_increase": pd.Series([np.nan] * N_YEARS),
                }
            },
            {
                "pp_coal_grid": [1000, 900, 800, 700, 600],
            },
            id="capacity_should_be_decreasing_by_100",
        ),
        pytest.param(
            {
                EE: pd.Series([0] * N_YEARS),
                HEAT: pd.Series([0] * N_YEARS),
            },
            {
                "pp_coal_grid": {
                    "unit_base_cap": 1000,
                    "unit_min_capacity": pd.Series([np.nan] * N_YEARS),
                    "unit_max_capacity": pd.Series([np.nan] * N_YEARS),
                    "unit_min_capacity_increase": pd.Series(
                        [np.nan] + [100] * (N_YEARS - 1)
                    ),
                    "unit_max_capacity_increase": pd.Series([np.nan] * N_YEARS),
                }
            },
            {
                "pp_coal_grid": [1000, 1100, 1200, 1300, 1400],
            },
            id="capacity_should_be_increasing_by_100",
        ),
        pytest.param(
            {
                EE: pd.Series([0] * N_YEARS),
                HEAT: pd.Series([0] * N_YEARS),
            },
            {
                "biomass_heat_plant_hs": {
                    "unit_base_cap": 1000,
                    "unit_min_capacity": pd.Series([np.nan] * N_YEARS),
                    "unit_max_capacity": pd.Series([np.nan] * N_YEARS),
                    "unit_min_capacity_increase": pd.Series(
                        [np.nan] + [100] * (N_YEARS - 1)
                    ),
                    "unit_max_capacity_increase": pd.Series([np.nan] * N_YEARS),
                },
                "pp_coal_grid": {
                    "unit_base_cap": 1000,
                    "unit_min_capacity": pd.Series([np.nan] + [1100] * (N_YEARS - 1)),
                    "unit_max_capacity": pd.Series([np.nan] + [1100] * (N_YEARS - 1)),
                    "unit_min_capacity_increase": pd.Series([np.nan] * N_YEARS),
                    "unit_max_capacity_increase": pd.Series([np.nan] * N_YEARS),
                },
            },
            {
                "biomass_heat_plant_hs": [1000, 1100, 1200, 1300, 1400],
                "pp_coal_grid": [1000, 1100, 1100, 1100, 1100],
            },
            id="capacity_should_be_increasing_by_100",
        ),
        pytest.param(
            {
                EE: pd.Series([10000000] * N_YEARS),
                HEAT: pd.Series([10000000] * N_YEARS),
            },
            {
                "pp_coal_grid": {
                    "unit_base_cap": 1000,
                    "unit_min_capacity": pd.Series([np.nan] * N_YEARS),
                    "unit_max_capacity": pd.Series([np.nan] + [1300] * (N_YEARS - 1)),
                    "unit_min_capacity_increase": pd.Series([np.nan] * N_YEARS),
                    "unit_max_capacity_increase": pd.Series(
                        [np.nan] + [100] * (N_YEARS - 1)
                    ),
                }
            },
            {
                "pp_coal_grid": [1000, 1100, 1200, 1300, 1300],
            },
            id="capacity_should_be_increasing_by_100_to_max_1300",
        ),
    ],
)
def test_generator_capacity_constraints(
    network: Network,
    aggr_yearly_energy_usage: dict[str, pd.Series],
    generators_params: dict[str, Any],
    excepted_capacity: dict[str, list[float]],
) -> None:
    opt_config = create_default_opt_config(
        hour_sample=np.arange(5),
        year_sample=np.arange(N_YEARS),
    )
    network.aggregated_consumers["aggr"].yearly_energy_usage = aggr_yearly_energy_usage
    set_network_elements_parameters(network.generators, generators_params)
    engine = run_opt_engine(network, opt_config)
    for name, expected in excepted_capacity.items():
        assert np.allclose(
            engine.results.generators_results.cap[name].iloc[:, 0].to_list(), expected
        )


@pytest.mark.parametrize(
    "generators_params",
    [
        pytest.param(
            {
                "pp_coal_grid": {
                    "unit_base_cap": 1000,
                    "unit_min_capacity": pd.Series([np.nan] + [1200] * (N_YEARS - 1)),
                    "unit_max_capacity": pd.Series([np.nan] + [1100] * (N_YEARS - 1)),
                    "unit_min_capacity_increase": pd.Series([np.nan] * N_YEARS),
                    "unit_max_capacity_increase": pd.Series([np.nan] * N_YEARS),
                }
            },
            id="min_capacity_should_be_greater_than_max_capacity",
        ),
        pytest.param(
            {
                "pp_coal_grid": {
                    "unit_base_cap": 1000,
                    "unit_min_capacity": pd.Series([np.nan] + [1100] * (N_YEARS - 1)),
                    "unit_max_capacity": pd.Series([np.nan] + [1100] * (N_YEARS - 1)),
                    "unit_min_capacity_increase": pd.Series(
                        [np.nan] + [100] * (N_YEARS - 1)
                    ),
                    "unit_max_capacity_increase": pd.Series(
                        [np.nan] + [100] * (N_YEARS - 1)
                    ),
                }
            },
            id="delta_cap_should_be_nan_when_cap_fixed",
        ),
    ],
)
def test_generator_capacity_bad_constraints(
    network: Network,
    generators_params: dict[str, Any],
) -> None:
    opt_config = create_default_opt_config(
        hour_sample=np.arange(5),
        year_sample=np.arange(N_YEARS),
    )
    set_network_elements_parameters(network.generators, generators_params)
    with pytest.raises(OptimizationError):
        run_opt_engine(network, opt_config).results  # noqa: F841
