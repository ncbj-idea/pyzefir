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
from tests.unit.optimization.gurobi.constants import N_YEARS
from tests.unit.optimization.gurobi.test_model.utils import (
    create_default_opf_config,
    run_opt_engine,
    set_network_elements_parameters,
)
from tests.unit.optimization.gurobi.utils import TOL


@pytest.mark.parametrize(
    ("hour_sample", "year_sample", "life_time", "build_time"),
    [
        (
            np.arange(100),
            np.arange(N_YEARS),
            9,
            0,
        ),
        (
            np.arange(100),
            np.arange(N_YEARS),
            7,
            0,
        ),
    ],
)
def test_local_supplementary_capacity_upper_bound_constraints(
    hour_sample: np.ndarray,
    year_sample: np.ndarray,
    life_time: int,
    build_time: int,
    network: Network,
) -> None:
    """
    Test local capacity evolution constraints
    """

    set_network_elements_parameters(
        network.aggregated_consumers,
        {
            "aggr": {
                "stack_base_fraction": {"lbs": 0.6, "lbs2": 0.4},
                "n_consumers": pd.Series([1000, 1000, 1000, 1000, 1000]),
            }
        },
    )
    set_network_elements_parameters(
        network.generator_types,
        {"local_coal_heat_plant": {"life_time": life_time, "build_time": build_time}},
    ),

    set_network_elements_parameters(
        network.generators,
        {"local_coal_heat_plant": {"min_device_nom_power": 24}},
    )

    # change generators such that will be 2 units of the same type
    set_network_elements_parameters(
        network.generators,
        {"local_coal_heat_plant2": {"energy_source_type": "local_coal_heat_plant"}},
    )

    opt_config = create_default_opf_config(hour_sample, year_sample)
    engine = run_opt_engine(network, opt_config)

    aggr_to_type = engine.indices.aggr_tgen_map
    t_type = list(aggr_to_type[0])[0]  # only one type in the test
    aggr_name = engine.indices.AGGR.mapping[0]
    u_idxs = _u_idxs(engine.parameters.gen.tgen, t_type)
    lt, bt = engine.parameters.tgen.lt[t_type], engine.parameters.tgen.bt[t_type]
    t_name, u_name = engine.indices.TGEN.mapping[t_type], engine.indices.GEN.mapping

    tcap = np.array(engine.results.generators_results.tcap[aggr_name][t_name][0])
    tcap_base_minus = np.array(
        engine.results.generators_results.tcap_base_minus["aggr"][t_name][0]
    )
    tcap_plus = np.array(
        engine.results.generators_results.tcap_plus[aggr_name][t_name][0]
    )
    tcap_minus = np.array(
        engine.results.generators_results.tcap_minus[aggr_name][t_name]
    )

    base_cap = sum(engine.parameters.gen.base_cap[u_idx] for u_idx in u_idxs)
    cap = engine.results.generators_results.cap

    sum_cap = sum(cap[u_name[u_idx]] for u_idx in u_idxs)
    sum_cap = np.array(sum_cap[0])
    assert np.allclose(tcap, sum_cap)

    t_all_cap_minus_sum = tcap_minus.sum(axis=1)
    assert np.all(tcap_plus + TOL >= t_all_cap_minus_sum)

    assert np.allclose(tcap, np.array([35000, 24000, 24000, 24000, 24000]))
    assert np.allclose(tcap_base_minus, np.array([0, 11000, 0, 0, 0]))
    assert tcap_plus.sum() <= TOL
    assert tcap_minus.sum() <= TOL

    for y in engine.indices.Y.ord:
        initial_cap = (
            base_cap - sum(tcap_base_minus[s] for s in range(1, y + 1)) if y < lt else 0
        )
        incr_cap = sum(tcap_plus[s] for s in _s_range(y, lt, bt))
        decr_cap = sum(
            tcap_minus[s, t]
            for s in _s_range(y, lt, bt)
            for t in _t_range(y, s, lt, bt)
        )
        assert abs(initial_cap + incr_cap - decr_cap - tcap[y]) <= TOL


def _u_idxs(t_gen: dict[int, int], t_type: int) -> set:
    return {u_idx for u_idx, u_type_idx in t_gen.items() if u_type_idx == t_type}


def _s_range(y: int, lt: int, bt: int) -> range:
    return range(max(0, y - lt - bt + 1), y - bt + 1)


def _t_range(y: int, s: int, lt: int, bt: int) -> range:
    return range(s + bt, min(y, s + bt + lt - 1) + 1)
