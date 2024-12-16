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
import pytest

from pyzefir.model.network import Network
from pyzefir.model.network_elements import DSR
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.test_model.utils import (
    set_network_elements_parameters,
)


@pytest.mark.parametrize(
    (
        "dsr",
        "expected_dsr",
        "dsr_bus",
        "expected_dsr_bus",
        "n_hours",
        "expected_compensation_periods",
    ),
    [
        (
            {
                "dsr_1": {
                    "compensation_factor": 0.1,
                    "balancing_period_len": 2,
                    "penalization": 1000,
                    "penalization_plus": 100,
                    "relative_shift_limit": None,
                    "abs_shift_limit": 0.6,
                }
            },
            {
                "abs_shift_limit": {0: 0.6},
                "balancing_period_len": {0: 2},
                "compensation_factor": {0: 0.1},
                "penalization": {0: 1000},
                "penalization_plus": {0: 100},
                "relative_shift_limit": {},
                "hourly_relative_shift_minus_limit": {0: 1.0},
                "hourly_relative_shift_plus_limit": {0: 1.0},
            },
            {"grid": "dsr_1"},
            {0: 0},
            5,
            {"dsr_1": [range(0, 2), range(2, 4), range(4, 5)]},
        ),
        (
            {
                "dsr_1": {
                    "compensation_factor": 0.1,
                    "balancing_period_len": 3,
                    "penalization": 1000,
                    "penalization_plus": 10,
                    "relative_shift_limit": 0.5,
                    "abs_shift_limit": None,
                    "hourly_relative_shift_minus_limit": 0.1,
                }
            },
            {
                "abs_shift_limit": {},
                "balancing_period_len": {0: 3},
                "compensation_factor": {0: 0.1},
                "penalization": {0: 1000},
                "penalization_plus": {0: 10},
                "relative_shift_limit": {0: 0.5},
                "hourly_relative_shift_minus_limit": {0: 0.1},
                "hourly_relative_shift_plus_limit": {0: 0.1},
            },
            {"grid": "dsr_1"},
            {0: 0},
            6,
            {"dsr_1": [range(0, 3), range(3, 6)]},
        ),
        (
            {
                "dsr_1": {
                    "compensation_factor": 0.1,
                    "balancing_period_len": 5,
                    "penalization": 1000,
                    "penalization_plus": -1000,
                    "relative_shift_limit": None,
                    "abs_shift_limit": None,
                    "hourly_relative_shift_minus_limit": 0.1,
                    "hourly_relative_shift_plus_limit": 0.05,
                }
            },
            {
                "abs_shift_limit": {},
                "balancing_period_len": {0: 5},
                "compensation_factor": {0: 0.1},
                "penalization": {0: 1000},
                "penalization_plus": {0: -1000},
                "relative_shift_limit": {},
                "hourly_relative_shift_minus_limit": {0: 0.1},
                "hourly_relative_shift_plus_limit": {0: 0.05},
            },
            {"grid": "dsr_1"},
            {0: 0},
            5,
            {"dsr_1": [range(5)]},
        ),
        (
            {
                "dsr_1": {
                    "compensation_factor": 0.1,
                    "balancing_period_len": 2,
                    "penalization": 1000,
                    "penalization_plus": 100,
                    "relative_shift_limit": 0.5,
                    "abs_shift_limit": 0.6,
                }
            },
            {
                "abs_shift_limit": {0: 0.6},
                "balancing_period_len": {0: 2},
                "compensation_factor": {0: 0.1},
                "penalization": {0: 1000},
                "penalization_plus": {0: 100},
                "relative_shift_limit": {0: 0.5},
                "hourly_relative_shift_minus_limit": {},
                "hourly_relative_shift_plus_limit": {},
            },
            {"grid": "dsr_1"},
            {0: 0},
            6,
            {"dsr_1": [range(0, 2), range(2, 4), range(4, 6)]},
        ),
        (
            {
                "dsr_1": {
                    "compensation_factor": 0.1,
                    "balancing_period_len": 2,
                    "penalization": 1000,
                    "penalization_plus": 200,
                    "relative_shift_limit": 0.5,
                    "abs_shift_limit": 0.6,
                    "hourly_relative_shift_minus_limit": 0.1,
                },
                "dsr_2": {
                    "compensation_factor": 0.2,
                    "balancing_period_len": 3,
                    "penalization": 100,
                    "penalization_plus": 14,
                    "relative_shift_limit": 0.4,
                    "abs_shift_limit": 0.7,
                    "hourly_relative_shift_plus_limit": 0.07,
                },
            },
            {
                "abs_shift_limit": {0: 0.6, 1: 0.7},
                "balancing_period_len": {0: 2, 1: 3},
                "compensation_factor": {0: 0.1, 1: 0.2},
                "penalization": {0: 1000, 1: 100},
                "penalization_plus": {0: 200, 1: 14},
                "relative_shift_limit": {0: 0.5, 1: 0.4},
                "hourly_relative_shift_minus_limit": {0: 0.1, 1: 1.0},
                "hourly_relative_shift_plus_limit": {0: 1.0, 1: 0.07},
            },
            {"grid": "dsr_2", "hs": "dsr_1"},
            {0: 1, 1: 0},
            5,
            {
                "dsr_1": [range(0, 2), range(2, 4), range(4, 5)],
                "dsr_2": [range(0, 3), range(3, 5)],
            },
        ),
        (
            {
                "dsr_1": {
                    "compensation_factor": 0.1,
                    "balancing_period_len": 2,
                    "penalization": 1000,
                    "penalization_plus": 500,
                    "relative_shift_limit": None,
                    "abs_shift_limit": 0.6,
                    "hourly_relative_shift_minus_limit": 0.1,
                    "hourly_relative_shift_plus_limit": 0.07,
                },
                "dsr_2": {
                    "compensation_factor": 0.2,
                    "balancing_period_len": 4,
                    "penalization": 100,
                    "penalization_plus": -50,
                    "relative_shift_limit": 0.4,
                    "abs_shift_limit": None,
                    "hourly_relative_shift_minus_limit": 0.2,
                    "hourly_relative_shift_plus_limit": 0.04,
                },
            },
            {
                "abs_shift_limit": {0: 0.6},
                "balancing_period_len": {0: 2, 1: 4},
                "compensation_factor": {0: 0.1, 1: 0.2},
                "penalization": {0: 1000, 1: 100},
                "penalization_plus": {0: 500, 1: -50},
                "relative_shift_limit": {1: 0.4},
                "hourly_relative_shift_minus_limit": {0: 0.1, 1: 0.2},
                "hourly_relative_shift_plus_limit": {0: 0.07, 1: 0.04},
            },
            {"grid": "dsr_2", "hs": "dsr_1"},
            {0: 1, 1: 0},
            5,
            {
                "dsr_1": [range(0, 2), range(2, 4), range(4, 5)],
                "dsr_2": [range(0, 4), range(4, 5)],
            },
        ),
    ],
)
def test_dsr_parameters(
    dsr: dict[str, dict[str, float]],
    expected_dsr: dict[str, dict[int, float]],
    dsr_bus: dict[str, str],
    expected_dsr_bus: dict[int, str],
    n_hours: int,
    expected_compensation_periods: list[range],
    complete_network: Network,
    opt_config: OptConfig,
) -> None:
    for dsr_name, dsr_content in dsr.items():
        complete_network.dsr.update(
            {
                dsr_name: DSR(
                    name=dsr_name,
                    compensation_factor=dsr_content["compensation_factor"],
                    balancing_period_len=int(dsr_content["balancing_period_len"]),
                    penalization_minus=dsr_content["penalization"],
                    penalization_plus=dsr_content["penalization_plus"],
                    relative_shift_limit=dsr_content["relative_shift_limit"],
                    abs_shift_limit=dsr_content["abs_shift_limit"],
                )
            }
        )
    for bus, dsr_name_el in dsr_bus.items():
        set_network_elements_parameters(
            complete_network.buses,
            {bus: {"dsr_type": dsr_name_el}},
        )
    opt_config.hour_sample = opt_config.hour_sample[:n_hours]
    opt_config.hours = opt_config.hours[:n_hours]
    indices = Indices(complete_network, opt_config)
    params = OptimizationParameters(complete_network, indices, opt_config)
    dsr_params = params.dsr
    assert dsr_params.abs_shift_limit == expected_dsr["abs_shift_limit"]
    assert dsr_params.balancing_period_len == expected_dsr["balancing_period_len"]
    assert dsr_params.penalization_minus == expected_dsr["penalization"]
    assert dsr_params.relative_shift_limit == expected_dsr["relative_shift_limit"]
    assert dsr_params.abs_shift_limit == expected_dsr["abs_shift_limit"]
    for name in indices.DSR.ii:
        assert (
            dsr_params.balancing_periods[indices.DSR.inverse[name]]
            == expected_compensation_periods[name]
        )
    assert params.bus.dsr_type == expected_dsr_bus
