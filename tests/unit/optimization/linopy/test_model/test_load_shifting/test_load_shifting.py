# PyZefir
# Copyright (C) 2024 Narodowe Centrum Badań Jądrowych
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
from pyzefir.model.network_elements import DSR
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.results import Results
from tests.unit.optimization.linopy.constants import N_YEARS
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opt_config,
    run_opt_engine,
    set_network_elements_parameters,
)
from tests.unit.optimization.linopy.test_model.utils_for_objective_testing.utils_for_obj_capex import (
    objective_capex,
)
from tests.unit.optimization.linopy.test_model.utils_for_objective_testing.utils_for_obj_dsr import (
    objective_dsr,
)
from tests.unit.optimization.linopy.test_model.utils_for_objective_testing.utils_for_obj_opex import (
    objective_opex,
)
from tests.unit.optimization.linopy.test_model.utils_for_objective_testing.utils_for_obj_varcost import (
    objective_varcost,
)
from tests.unit.optimization.linopy.utils import TOL


@pytest.mark.parametrize(
    ("dsr", "n_consumers", "hour_sample", "n_years_aggregation"),
    [
        (
            {
                "dsr_1": {
                    "bus": "local_ee_bus",
                    "balancing_period_len": 20,
                    "compensation_factor": 1,
                    "penalization": 5,
                    "penalization_plus": -10,
                    "abs_shift_limit": 500,
                    "relative_shift_limit": 0.5,
                    "hourly_relative_shift_minus_limit": 0.5,
                    "hourly_relative_shift_plus_limit": 0.55,
                }
            },
            pd.Series([10, 10, 10, 10, 10]),
            np.arange(100),
            1,
        ),
        (
            {
                "dsr_1": {
                    "bus": "local_ee_bus",
                    "balancing_period_len": 20,
                    "compensation_factor": 0.8,
                    "penalization": 5,
                    "penalization_plus": 0,
                    "abs_shift_limit": 0.5,
                    "relative_shift_limit": 0.5,
                    "hourly_relative_shift_minus_limit": 0.3,
                    "hourly_relative_shift_plus_limit": 0.1,
                }
            },
            pd.Series([10, 10, 10, 10, 10]),
            np.arange(100),
            1,
        ),
        (
            {
                "dsr_1": {
                    "bus": "local_ee_bus",
                    "balancing_period_len": 20,
                    "compensation_factor": 0.1,
                    "penalization": 5,
                    "penalization_plus": 0,
                    "abs_shift_limit": 0.5,
                    "relative_shift_limit": None,
                    "hourly_relative_shift_minus_limit": 0.4,
                    "hourly_relative_shift_plus_limit": 0.2,
                }
            },
            pd.Series([10, 10, 10, 10, 10]),
            np.arange(100),
            1,
        ),
        (
            {
                "dsr_1": {
                    "bus": "local_ee_bus",
                    "balancing_period_len": 20,
                    "compensation_factor": 0.1,
                    "penalization": 5,
                    "penalization_plus": 0,
                    "abs_shift_limit": None,
                    "relative_shift_limit": 0.7,
                    "hourly_relative_shift_minus_limit": 0.0,
                    "hourly_relative_shift_plus_limit": None,
                }
            },
            pd.Series([10, 10, 10, 10, 10]),
            np.arange(100),
            1,
        ),
        (
            {
                "dsr_1": {
                    "bus": "local_ee_bus",
                    "balancing_period_len": 20,
                    "compensation_factor": 0.1,
                    "penalization": 5,
                    "penalization_plus": 2,
                    "abs_shift_limit": None,
                    "relative_shift_limit": None,
                    "hourly_relative_shift_minus_limit": 0.3,
                    "hourly_relative_shift_plus_limit": 0.7,
                }
            },
            pd.Series([10, 10, 10, 10, 10]),
            np.arange(100),
            1,
        ),
        (
            {
                "dsr_1": {
                    "bus": "local_ee_bus",
                    "balancing_period_len": 20,
                    "compensation_factor": 1,
                    "penalization": 10,
                    "penalization_plus": 50,
                    "abs_shift_limit": 0.5,
                    "relative_shift_limit": 0.5,
                    "hourly_relative_shift_minus_limit": 0.3,
                    "hourly_relative_shift_plus_limit": None,
                }
            },
            pd.Series([10, 30, 10, 15, 50]),
            np.arange(50),
            1,
        ),
        (
            {
                "dsr_1": {
                    "bus": "local_ee_bus",
                    "balancing_period_len": 10,
                    "compensation_factor": 0.5,
                    "penalization": 5,
                    "penalization_plus": 1,
                    "abs_shift_limit": 50,
                    "relative_shift_limit": 0.4,
                    "hourly_relative_shift_minus_limit": 0.9,
                    "hourly_relative_shift_plus_limit": 0.9,
                },
                "dsr_2": {
                    "bus": "local_ee_bus2",
                    "balancing_period_len": 12,
                    "compensation_factor": 0.7,
                    "penalization": 4,
                    "penalization_plus": 2,
                    "abs_shift_limit": 40,
                    "relative_shift_limit": 1,
                    "hourly_relative_shift_minus_limit": 0.9,
                    "hourly_relative_shift_plus_limit": 0.9,
                },
            },
            pd.Series([100, 20, 30, 10, 100]),
            np.arange(50),
            1,
        ),
        (
            {
                "dsr_1": {
                    "bus": "local_ee_bus",
                    "balancing_period_len": 25,
                    "compensation_factor": 0.5,
                    "penalization": 5,
                    "penalization_plus": 3,
                    "abs_shift_limit": 50,
                    "relative_shift_limit": 0.4,
                    "hourly_relative_shift_minus_limit": 0.39,
                    "hourly_relative_shift_plus_limit": 0.41,
                },
                "dsr_2": {
                    "bus": "local_ee_bus2",
                    "balancing_period_len": 25,
                    "compensation_factor": 0.7,
                    "penalization": 4,
                    "penalization_plus": 4,
                    "abs_shift_limit": 40,
                    "relative_shift_limit": 0.5,
                    "hourly_relative_shift_minus_limit": 0.41,
                    "hourly_relative_shift_plus_limit": 0.39,
                },
            },
            pd.Series([100, 20, 30, 10, 100]),
            np.arange(20),
            1,
        ),
        (
            {
                "dsr_1": {
                    "bus": "local_ee_bus",
                    "balancing_period_len": 25,
                    "compensation_factor": 0.5,
                    "penalization": 5,
                    "penalization_plus": 0,
                    "abs_shift_limit": 50,
                    "relative_shift_limit": 0.4,
                    "hourly_relative_shift_minus_limit": 0.1,
                    "hourly_relative_shift_plus_limit": 0.2,
                },
                "dsr_2": {
                    "bus": "local_ee_bus2",
                    "balancing_period_len": 25,
                    "compensation_factor": 0.7,
                    "penalization": 4,
                    "penalization_plus": 0,
                    "abs_shift_limit": 40,
                    "relative_shift_limit": None,
                    "hourly_relative_shift_minus_limit": 0.2,
                    "hourly_relative_shift_plus_limit": None,
                },
            },
            pd.Series([100, 20, 30, 10, 100]),
            np.arange(20),
            1,
        ),
        (
            {
                "dsr_1": {
                    "bus": "local_ee_bus",
                    "balancing_period_len": 25,
                    "compensation_factor": 0.5,
                    "penalization": 5,
                    "penalization_plus": 2,
                    "abs_shift_limit": None,
                    "relative_shift_limit": 0.4,
                    "hourly_relative_shift_minus_limit": 0.2,
                    "hourly_relative_shift_plus_limit": None,
                },
                "dsr_2": {
                    "bus": "local_ee_bus2",
                    "balancing_period_len": 25,
                    "compensation_factor": 0.7,
                    "penalization": 4,
                    "penalization_plus": 10,
                    "abs_shift_limit": None,
                    "relative_shift_limit": None,
                    "hourly_relative_shift_minus_limit": 0.2,
                    "hourly_relative_shift_plus_limit": None,
                },
            },
            pd.Series([100, 20, 30, 10, 100]),
            np.arange(20),
            1,
        ),
        (
            {
                "dsr_1": {
                    "bus": "local_ee_bus",
                    "balancing_period_len": 25,
                    "compensation_factor": 0.5,
                    "penalization": 5,
                    "penalization_plus": 2,
                    "abs_shift_limit": None,
                    "relative_shift_limit": 0.4,
                    "hourly_relative_shift_minus_limit": 1.0,
                    "hourly_relative_shift_plus_limit": 0.9,
                },
                "dsr_2": {
                    "bus": "local_ee_bus2",
                    "balancing_period_len": 25,
                    "compensation_factor": 0.7,
                    "penalization": 4,
                    "penalization_plus": 3,
                    "abs_shift_limit": None,
                    "relative_shift_limit": None,
                    "hourly_relative_shift_minus_limit": 1.0,
                    "hourly_relative_shift_plus_limit": None,
                },
            },
            pd.Series([100, 20, 30, 10, 100]),
            np.arange(20),
            3,
        ),
        (
            {
                "dsr_1": {
                    "bus": "local_ee_bus",
                    "balancing_period_len": 20,
                    "compensation_factor": 0.1,
                    "penalization": 5,
                    "penalization_plus": 4,
                    "abs_shift_limit": 0.5,
                    "relative_shift_limit": None,
                    "hourly_relative_shift_minus_limit": 0.0,
                    "hourly_relative_shift_plus_limit": None,
                }
            },
            pd.Series([10, 10, 10, 10, 10]),
            np.arange(100),
            6,
        ),
    ],
)
def test_load_shifting(
    dsr: dict[str, dict[str, str | int | float]],
    n_consumers: pd.Series,
    hour_sample: np.ndarray,
    n_years_aggregation: int,
    network: Network,
) -> None:
    """
    test dsr constraints and sampling (load shifting)
    test objective
    no transission fee and ens in the network
    """

    for dsr_name, dsr_params in dsr.items():
        set_network_elements_parameters(
            network.buses, {dsr_params["bus"]: {"dsr_type": dsr_name}}
        )
        network.dsr.update(
            {
                dsr_name: DSR(
                    name=dsr_name,
                    balancing_period_len=dsr_params["balancing_period_len"],
                    compensation_factor=dsr_params["compensation_factor"],
                    penalization_minus=dsr_params["penalization"],
                    penalization_plus=dsr_params["penalization_plus"],
                    abs_shift_limit=dsr_params["abs_shift_limit"],
                    relative_shift_limit=dsr_params["relative_shift_limit"],
                    hourly_relative_shift_minus_limit=dsr_params[
                        "hourly_relative_shift_minus_limit"
                    ],
                    hourly_relative_shift_plus_limit=dsr_params[
                        "hourly_relative_shift_plus_limit"
                    ],
                )
            }
        )

    set_network_elements_parameters(
        network.aggregated_consumers,
        {
            "aggr": {
                "stack_base_fraction": {"lbs": 0.7, "lbs2": 0.3},
                "n_consumers": n_consumers,
            }
        },
    )

    set_network_elements_parameters(
        network.generators,
        {
            "local_coal_heat_plant": {"min_device_nom_power": 60},
            "local_coal_heat_plant2": {"min_device_nom_power": 15},
        },
    )

    set_network_elements_parameters(
        network.lines,
        {
            "grid->local_ee_bus": {"transmission_fee": None},
            "grid->local_ee_bus2": {"transmission_fee": None},
        },
    )

    opt_config = create_default_opt_config(
        hour_sample,
        np.arange(N_YEARS),
        year_aggregates=np.array([1] + [n_years_aggregation] * (N_YEARS - 2) + [1]),
    )
    engine = run_opt_engine(network, opt_config)

    parameters = engine.parameters
    results = engine.results
    indices = engine.indices

    shift_plus = results.bus_results.shift_plus
    shift_minus = results.bus_results.shift_minus
    compensation_factor = parameters.dsr.compensation_factor
    relative_shift_limit = parameters.dsr.relative_shift_limit
    abs_shift_limit = parameters.dsr.abs_shift_limit
    hourly_relative_shift_plus_limit = parameters.dsr.hourly_relative_shift_plus_limit
    hourly_relative_shift_minus_limit = parameters.dsr.hourly_relative_shift_minus_limit

    _test_objective(indices, parameters, results)

    for bus_idx, dsr_idx in engine.parameters.bus.dsr_type.items():
        if bus_idx in engine.parameters.bus.lbs_mapping:
            intervals = list(parameters.dsr.balancing_periods[dsr_idx])
            net_load = np.asarray(_bus_net_load(parameters, results, indices, bus_idx))
            bus_name = indices.BUS.mapping[bus_idx]
            for interval in intervals:
                for y in indices.Y.ord:
                    _test_compensation_constraints(
                        y,
                        shift_plus[bus_name],
                        shift_minus[bus_name],
                        interval,
                        compensation_factor[dsr_idx],
                    )
                    if dsr_idx in relative_shift_limit:
                        _test_relative_shift_limit(
                            y,
                            shift_minus[bus_name],
                            interval,
                            relative_shift_limit[dsr_idx],
                            net_load,
                        )

                    if dsr_idx in hourly_relative_shift_plus_limit:
                        _test_relative_shift_limit_hourly(
                            y,
                            shift_plus[bus_name],
                            hourly_relative_shift_plus_limit[dsr_idx],
                            net_load,
                        )

                    _test_relative_shift_limit_hourly(
                        y,
                        shift_minus[bus_name],
                        hourly_relative_shift_minus_limit[dsr_idx],
                        net_load,
                    )

                    if dsr_idx in abs_shift_limit.keys():
                        _test_absolute_shift_limit(
                            y, shift_minus[bus_name], interval, abs_shift_limit[dsr_idx]
                        )


def _test_objective(
    indices: Indices, parameters: OptimizationParameters, results: Results
) -> None:
    expected_obj = objective_capex(indices, parameters, results)
    expected_obj += objective_opex(indices, parameters, results)
    expected_obj += objective_varcost(indices, parameters, results)
    expected_obj += objective_dsr(indices, parameters, results)
    assert abs(results.objective_value - expected_obj) < TOL


def _test_compensation_constraints(
    y: int,
    shift_plus: pd.DataFrame,
    shift_minus: pd.DataFrame,
    interval: list[int],
    compensation_factor: float,
) -> None:
    vec_lhs = np.array([shift_plus[y][h] for h in interval])
    vec_rhs = compensation_factor * np.array([shift_minus[y][h] for h in interval])
    assert np.isclose(vec_lhs.sum(), vec_rhs.sum(), atol=TOL)


def _test_relative_shift_limit(
    y: int,
    shift_minus: pd.DataFrame,
    interval: list[int],
    relative_shift_limit: float,
    net_load: np.ndarray,
) -> None:
    lhs = sum(shift_minus[y][h] for h in interval)
    rhs = relative_shift_limit * sum(net_load[h, y] for h in interval)
    assert bool(lhs <= rhs + TOL)


def _test_relative_shift_limit_hourly(
    y: int,
    shift_var: pd.DataFrame,
    relative_shift_limit_hourly: float,
    net_load: np.ndarray,
) -> None:
    lhs = shift_var[y].values
    rhs = relative_shift_limit_hourly * net_load[:, y]
    assert np.all(lhs <= rhs + TOL)


def _test_absolute_shift_limit(
    y: int, shift_minus: pd.DataFrame, interval: list[int], abs_shift_limit: float
) -> None:
    lhs = sum(shift_minus[y][h] for h in interval)
    rhs = abs_shift_limit
    assert bool(lhs <= rhs + TOL)


def _bus_net_load(
    parameters: OptimizationParameters, results: Results, indices: Indices, bus_idx: int
) -> np.ndarray | float:
    return _fraction_demnd(parameters, results, indices, bus_idx) + _conversion_rate(
        parameters, results, indices, bus_idx
    )


def _conversion_rate(
    parameters: OptimizationParameters, results: Results, indices: Indices, bus_idx: int
) -> np.ndarray | float:
    bus_et, result = parameters.bus.et[bus_idx], 0.0
    for gen_idx in parameters.bus.generators[bus_idx]:
        if bus_et in parameters.gen.conv_rate[gen_idx]:
            gen_name = indices.GEN.mapping[gen_idx]
            result += results.generators_results.gen[
                gen_name
            ] / parameters.gen.conv_rate[gen_idx][bus_et].reshape(-1, 1)
    return result


def _fraction_demnd(
    parameters: OptimizationParameters, results: Results, indices: Indices, bus_idx: int
) -> np.ndarray | float:
    if bus_idx not in parameters.bus.lbs_mapping:
        return 0.0

    lbs_idx, energy_type = (
        parameters.bus.lbs_mapping[bus_idx],
        parameters.bus.et[bus_idx],
    )
    aggr_idx = parameters.lbs.aggr_idx[lbs_idx]
    aggr_name = indices.AGGR.mapping[aggr_idx]
    lbs_name = indices.LBS.mapping[lbs_idx]
    dem, frac = (
        parameters.aggr.dem[aggr_idx][energy_type],
        results.fractions_results.frac[aggr_name][lbs_name],
    )
    frac = frac["frac"].values

    return dem * frac


def _shift(
    parameters: OptimizationParameters,
    results: Results,
    indices: Indices,
    bus_name: str,
) -> np.ndarray | float:
    return (
        results.bus_results.shift_plus[bus_name]
        - results.bus_results.shift_minus[bus_name]
        if indices.DSR.inverse[bus_name] in parameters.bus.dsr_type
        else 0.0
    )
