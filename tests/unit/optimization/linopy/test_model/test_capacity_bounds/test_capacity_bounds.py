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
from pyzefir.model.network_elements import CapacityBound
from tests.unit.optimization.linopy.constants import N_YEARS
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opt_config,
    run_opt_engine,
    set_network_elements_parameters,
)
from tests.unit.optimization.linopy.utils import TOL


@pytest.mark.parametrize(
    ("capacity_bound", "hour_sample"),
    [
        (
            [
                {
                    "name": "name",
                    "left_technology": "local_pv",
                    "left_coefficient": 0.3,
                    "right_technology": "pp_coal_grid",
                    "sense": "EQ",
                },
                {
                    "name": "name2",
                    "left_technology": "biomass_heat_plant_hs",
                    "left_coefficient": 0.4,
                    "right_technology": "coal_heat_plant_hs",
                    "sense": "EQ",
                },
            ],
            np.arange(30),
        ),
        (
            [
                {
                    "name": "name",
                    "left_technology": "local_pv",
                    "left_coefficient": 0.3,
                    "right_technology": "pp_coal_grid",
                    "sense": "EQ",
                },
                {
                    "name": "name2",
                    "left_technology": "biomass_heat_plant_hs",
                    "left_coefficient": 0.4,
                    "right_technology": "coal_heat_plant_hs",
                    "sense": "LEQ",
                },
            ],
            np.arange(30),
        ),
        (
            [
                {
                    "name": "name",
                    "left_technology": "local_pv",
                    "left_coefficient": 0.3,
                    "right_technology": "pp_coal_grid",
                    "sense": "LEQ",
                },
                {
                    "name": "name2",
                    "left_technology": "biomass_heat_plant_hs",
                    "left_coefficient": 0.4,
                    "right_technology": "coal_heat_plant_hs",
                    "sense": "LEQ",
                },
            ],
            np.arange(30),
        ),
        ([], np.arange(30)),
        (
            [
                {
                    "name": "name",
                    "left_technology": "ee_storage",
                    "left_coefficient": 0.7,
                    "right_technology": "heat_storage",
                    "sense": "EQ",
                }
            ],
            np.arange(50),
        ),
        (
            [
                {
                    "name": "name",
                    "left_technology": "ee_storage",
                    "left_coefficient": 0.7,
                    "right_technology": "heat_storage",
                    "sense": "LEQ",
                }
            ],
            np.arange(50),
        ),
        (
            [
                {
                    "name": "name",
                    "left_technology": "local_pv",
                    "left_coefficient": 0.9,
                    "right_technology": "heat_storage",
                    "sense": "EQ",
                }
            ],
            np.arange(50),
        ),
        (
            [
                {
                    "name": "name",
                    "left_technology": "local_pv",
                    "left_coefficient": 0.9,
                    "right_technology": "heat_storage",
                    "sense": "LEQ",
                }
            ],
            np.arange(50),
        ),
        (
            [
                {
                    "name": "name",
                    "left_technology": "heat_storage",
                    "left_coefficient": 0.9,
                    "right_technology": "local_pv",
                    "sense": "EQ",
                }
            ],
            np.arange(50),
        ),
        (
            [
                {
                    "name": "name",
                    "left_technology": "heat_storage",
                    "left_coefficient": 0.9,
                    "right_technology": "local_pv",
                    "sense": "LEQ",
                }
            ],
            np.arange(50),
        ),
        (
            [
                {
                    "name": "name",
                    "left_technology": "heat_storage",
                    "left_coefficient": 0.9,
                    "right_technology": "local_pv",
                    "sense": "LEQ",
                },
                {
                    "name": "name2",
                    "left_technology": "local_pv",
                    "left_coefficient": 0.4,
                    "right_technology": "pp_coal_grid",
                    "sense": "EQ",
                },
            ],
            np.arange(50),
        ),
        (
            [
                {
                    "name": "name",
                    "left_technology": "heat_storage",
                    "left_coefficient": 0.8,
                    "right_technology": "local_pv",
                    "sense": "EQ",
                },
                {
                    "name": "name2",
                    "left_technology": "local_pv",
                    "left_coefficient": 0.5,
                    "right_technology": "pp_coal_grid",
                    "sense": "EQ",
                },
            ],
            np.arange(50),
        ),
    ],
)
def test_capacity_bound(
    capacity_bound: dict,
    hour_sample: np.ndarray,
    network: Network,
) -> None:
    """
    Test capacity bounds constraints: for generators and storages
    """

    set_network_elements_parameters(
        network.aggregated_consumers,
        {
            "aggr": {
                "n_consumers": pd.Series([1000, 1200, 1400, 1600, 2000]),
            },
        },
    )
    set_network_elements_parameters(
        network.storage_types,
        {
            "ee_storage_type": {
                "capex": np.array([70, 60, 55, 40, 15]),
                "opex": np.array([30, 20, 10, 5, 5]),
            }
        },
    )
    set_network_elements_parameters(
        network.storage_types,
        {
            "heat_storage_type": {
                "capex": np.array([50, 50, 55, 50, 55]),
                "opex": np.array([30, 30, 30, 30, 35]),
            }
        },
    )
    for cap_bound in capacity_bound:
        network.add_capacity_bound(
            CapacityBound(
                name=cap_bound["name"],
                left_coefficient=cap_bound["left_coefficient"],
                left_technology=cap_bound["left_technology"],
                right_technology=cap_bound["right_technology"],
                sense=cap_bound["sense"],
            )
        )
    opt_config = create_default_opt_config(hour_sample, np.arange(N_YEARS))
    engine = run_opt_engine(network, opt_config)
    gen_cap, stor_cap = (
        engine.results.generators_results.cap,
        engine.results.storages_results.cap,
    )
    capacity_bounds = engine.parameters.capacity_bounds
    gen_map, stor_map = engine.indices.GEN.mapping, engine.indices.STOR.mapping
    for idx in engine.indices.CAP_BOUND.ord:
        lhs_idx, lhs_type, rhs_idx, rhs_type, sense, coeff = (
            capacity_bounds.lhs_idx[idx],
            capacity_bounds.lhs_type[idx],
            capacity_bounds.rhs_idx[idx],
            capacity_bounds.rhs_type[idx],
            capacity_bounds.sense[idx],
            capacity_bounds.coeff[idx],
        )
        left_hs = (
            np.asarray(gen_cap[gen_map[lhs_idx]].cap)[1:] * coeff
            if lhs_type == "GEN"
            else np.asarray(stor_cap[stor_map[lhs_idx]].cap)[1:] * coeff
        )
        right_hs = (
            np.asarray(gen_cap[gen_map[rhs_idx]].cap)[1:]
            if rhs_type == "GEN"
            else np.asarray(stor_cap[stor_map[rhs_idx]].cap)[1:]
        )
        assert (
            np.allclose(left_hs, right_hs)
            if sense == "EQ"
            else np.all(left_hs <= right_hs + TOL)
        )
