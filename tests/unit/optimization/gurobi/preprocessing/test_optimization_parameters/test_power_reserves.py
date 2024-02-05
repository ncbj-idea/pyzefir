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
from pyzefir.model.utils import NetworkConstants
from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.opt_config import OptConfig


@pytest.mark.parametrize(
    ("gen_tags", "power_reserves", "expected_result"),
    [
        (
            {"pp_coal_grid": ["tag1"]},
            {"power_reserves": {"electricity": {"tag1": 50.0}}},
            {"electricity": {0: 50.0}},
        )
    ],
)
def test_create_power_reserves(
    gen_tags: dict[str, list[str]],
    power_reserves: dict[str, dict[str, float]],
    expected_result: dict[str, dict[int, float]],
    complete_network: Network,
    opt_config: OptConfig,
) -> None:
    for generator_name, tags in gen_tags.items():
        complete_network.generators[generator_name].tags = tags

    indices = Indices(complete_network, opt_config)

    constants = complete_network.constants.__dict__
    complete_network.constants = NetworkConstants(**constants | power_reserves)

    result = OptimizationParameters(
        complete_network, indices, opt_config
    ).scenario_parameters
    assert result.power_reserves == expected_result
