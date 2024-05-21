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
from pyzefir.optimization.results import Results
from tests.unit.optimization.linopy.constants import N_YEARS
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opf_config,
    run_opt_engine,
    set_network_elements_parameters,
)


@pytest.mark.parametrize(
    ("capacity_bindings",),
    [
        (
            {
                "local_coal_heat_plant": "local_heat",
                "local_coal_heat_plant2": "local_heat",
                "local_pv": "pv",
                "local_pv2": "pv",
            },
        ),
        (
            {
                "local_coal_heat_plant": "group2",
                "local_coal_heat_plant2": "group2",
                "chp_coal_grid_hs": "group2",
                "local_pv": "group1",
                "local_pv2": "group1",
                "pp_coal_grid": "group1",
            },
        ),
        (
            {
                "local_coal_heat_plant": "local_heat",
                "local_coal_heat_plant2": "local_heat",
            },
        ),
        (
            {
                "local_coal_heat_plant": "local_heat",
                "local_coal_heat_plant2": "local_heat2",
            },
        ),
        ({"local_coal_heat_plant": "local_heat"},),
    ],
)
def test_engine_capacity_bindings(
    capacity_bindings: dict[str, str],
    network: Network,
) -> None:
    """
    test capacity bindings
    """

    set_network_elements_parameters(
        network.aggregated_consumers,
        {
            "aggr": {
                "stack_base_fraction": {
                    "lbs": 0.3,
                    "lbs2": 0.7,
                },
                "n_consumers": pd.Series([30000] * N_YEARS),
            }
        },
    )
    opt_config = create_default_opf_config(np.arange(50), N_YEARS)

    unit_min_start = pd.Series([1] * N_YEARS)
    iter = 0
    for gen_data in network.generators.values():
        gen_data.unit_min_capacity = unit_min_start + iter
        iter += 1

    for gen_name, binding in capacity_bindings.items():
        network.generators[gen_name].generator_binding = binding

    engine = run_opt_engine(network, opt_config)
    _binding_test(engine.results, capacity_bindings)


def _binding_test(results: Results, capacity_bindings: dict[str, str]) -> None:
    binding_markers = set(capacity_bindings.values())
    res_cap = results.generators_results.cap
    for binding_marker in binding_markers:
        gen_names = [
            gen_name
            for gen_name, binding in capacity_bindings.items()
            if binding == binding_marker
        ]
        if len(gen_names) >= 2:
            ref_gen_name = gen_names[0]
            ref_cap = np.asarray(res_cap[ref_gen_name].values.T[0][1:])
            rest_gen_names = gen_names[1:]
            for rest_gen_name in rest_gen_names:
                rest_cap = np.asarray(res_cap[rest_gen_name].values.T[0][1:])
                assert np.all(np.isclose(ref_cap, rest_cap))
