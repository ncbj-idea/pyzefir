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

from pyzefir.model.network import Network
from pyzefir.optimization.gurobi.preprocessing.indices import Indices
from pyzefir.optimization.gurobi.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.opt_config import OptConfig


def test_create(complete_network: Network, opt_config: OptConfig) -> None:
    indices = Indices(complete_network, opt_config)
    lines = complete_network.lines
    result = OptimizationParameters(complete_network, indices, opt_config).line

    for line_id, line_name in indices.LINE.mapping.items():
        line = lines[line_name]
        assert result.et[line_id] == line.energy_type
        assert result.cap[line_id] == line.max_capacity
        assert result.loss[line_id] == line.transmission_loss
        assert result.bus_from[line_id] == indices.BUS.inverse[line.fr]
        assert result.bus_to[line_id] == indices.BUS.inverse[line.to]
        if line.transmission_fee:
            assert result.tf[line_id] == indices.TF.inverse[line.transmission_fee]
