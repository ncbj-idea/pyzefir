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

import pytest

from pyzefir.model.network import Network
from pyzefir.model.network_elements import CapacityBound
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.opt_config import OptConfig


@pytest.mark.parametrize(
    ("capacity_bounds", "expected_capacity_bounds"),
    [
        (
            [
                {
                    "name": "name1",
                    "left_technology": "pp_coal_grid",
                    "left_coefficient": 0.9,
                    "right_technology": "heat_plant_biomass_hs",
                    "sense": "EQ",
                }
            ],
            {
                "coeff": {0: 0.9},
                "lhs_idx": {0: 2},
                "lhs_type": {0: "GEN"},
                "rhs_idx": {0: 5},
                "rhs_type": {0: "GEN"},
                "sense": {0: "EQ"},
            },
        ),
        (
            [
                {
                    "name": "name1",
                    "left_technology": "pp_coal_grid",
                    "left_coefficient": 0.9,
                    "right_technology": "heat_plant_biomass_hs",
                    "sense": "EQ",
                },
                {
                    "name": "name2",
                    "left_technology": "pp_gas_grid",
                    "left_coefficient": 0.8,
                    "right_technology": "heat_plant_coal_hs",
                    "sense": "EQ",
                },
            ],
            {
                "coeff": {0: 0.9, 1: 0.8},
                "lhs_idx": {0: 2, 1: 3},
                "lhs_type": {0: "GEN", 1: "GEN"},
                "rhs_idx": {0: 5, 1: 4},
                "rhs_type": {0: "GEN", 1: "GEN"},
                "sense": {0: "EQ", 1: "EQ"},
            },
        ),
        (
            [
                {
                    "name": "name3",
                    "left_technology": "heat_storage_hs",
                    "left_coefficient": 0.88,
                    "right_technology": "heat_storage_hs_2",
                    "sense": "EQ",
                }
            ],
            {
                "coeff": {0: 0.88},
                "lhs_idx": {0: 0},
                "lhs_type": {0: "STOR"},
                "rhs_idx": {0: 1},
                "rhs_type": {0: "STOR"},
                "sense": {0: "EQ"},
            },
        ),
        (
            [
                {
                    "name": "name4",
                    "left_technology": "pp_coal_grid",
                    "left_coefficient": 0.78,
                    "right_technology": "heat_storage_hs_2",
                    "sense": "LEQ",
                }
            ],
            {
                "coeff": {0: 0.78},
                "lhs_idx": {0: 2},
                "lhs_type": {0: "GEN"},
                "rhs_idx": {0: 1},
                "rhs_type": {0: "STOR"},
                "sense": {0: "LEQ"},
            },
        ),
        (
            [
                {
                    "name": "name5",
                    "left_technology": "heat_storage_hs",
                    "left_coefficient": 0.79,
                    "right_technology": "pp_coal_grid",
                    "sense": "LEQ",
                }
            ],
            {
                "coeff": {0: 0.79},
                "lhs_idx": {0: 0},
                "lhs_type": {0: "STOR"},
                "rhs_idx": {0: 2},
                "rhs_type": {0: "GEN"},
                "sense": {0: "LEQ"},
            },
        ),
    ],
)
def test_create_capacity_bound(
    capacity_bounds: list[dict],
    expected_capacity_bounds: dict[str, dict],
    complete_network: Network,
    opt_config: OptConfig,
) -> None:
    for cap_bound in capacity_bounds:
        complete_network.add_capacity_bound(
            CapacityBound(
                name=cap_bound["name"],
                left_coefficient=cap_bound["left_coefficient"],
                left_technology=cap_bound["left_technology"],
                right_technology=cap_bound["right_technology"],
                sense=cap_bound["sense"],
            )
        )
    indices = Indices(complete_network, opt_config)
    cap_bounds = OptimizationParameters(
        complete_network, indices, opt_config
    ).capacity_bounds
    assert cap_bounds.coeff == expected_capacity_bounds["coeff"]
    assert cap_bounds.lhs_idx == expected_capacity_bounds["lhs_idx"]
    assert cap_bounds.lhs_type == expected_capacity_bounds["lhs_type"]
    assert cap_bounds.rhs_idx == expected_capacity_bounds["rhs_idx"]
    assert cap_bounds.rhs_type == expected_capacity_bounds["rhs_type"]
    assert cap_bounds.sense == expected_capacity_bounds["sense"]
