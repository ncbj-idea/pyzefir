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
from pyzefir.model.utils import NetworkConstants
from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.gurobi.test_model.utils import (
    set_network_elements_parameters,
)


@pytest.mark.parametrize(
    ("min_max_gen_frac", "expected_values", "unit_tags"),
    [
        (
            {
                "min_generation_fraction": {"EE": {("test_tag", "test_tag2"): 0.1}},
                "max_generation_fraction": {"EE": {("test_tag", "test_tag2"): 0.9}},
            },
            {
                "min_generation_fraction": {"EE": {(0, 1): 0.1}},
                "max_generation_fraction": {"EE": {(0, 1): 0.9}},
            },
            {"pp_coal_grid": ["test_tag"], "pp_gas_grid": ["test_tag", "test_tag2"]},
        ),
        (
            {
                "min_generation_fraction": {"HEAT": {("test_tag", "test_tag2"): 0.3}},
                "max_generation_fraction": {"HEAT": {("test_tag", "test_tag2"): 0.7}},
            },
            {
                "min_generation_fraction": {"HEAT": {(0, 1): 0.3}},
                "max_generation_fraction": {"HEAT": {(0, 1): 0.7}},
            },
            {
                "chp_coal_grid_hs": ["test_tag"],
                "heat_pump_grid_hs": ["test_tag2"],
                "heat_plant_biomass_hs": ["test_tag2"],
            },
        ),
    ],
)
def test_create_min_max_generation_fractions(
    min_max_gen_frac: dict[str, dict[tuple[str, str], float]],
    expected_values: dict[str, dict[tuple[int, int], float]],
    unit_tags: dict[str, list[str]],
    complete_network: Network,
    opt_config: OptConfig,
) -> None:
    constants = complete_network.constants.__dict__
    complete_network.constants = NetworkConstants(**constants | min_max_gen_frac)
    for unit, unit_tag in unit_tags.items():
        set_network_elements_parameters(
            complete_network.generators, {unit: {"tags": unit_tag}}
        )

    indices = Indices(complete_network, opt_config)
    params = OptimizationParameters(
        complete_network, indices, opt_config
    ).scenario_parameters

    assert expected_values["min_generation_fraction"] == params.min_generation_fraction
    assert expected_values["max_generation_fraction"] == params.max_generation_fraction
