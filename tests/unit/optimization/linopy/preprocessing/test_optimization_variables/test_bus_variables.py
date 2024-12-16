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
from typing import Callable

import numpy as np
import pytest
from linopy import Model

from pyzefir.model.network import Network
from pyzefir.model.network_elements import DSR
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.linopy.preprocessing.variables.bus_variables import (
    BusVariables,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.names import EE, HEAT
from tests.unit.optimization.linopy.preprocessing.conftest import lbs_bus_name


def _add_dsr_to_network(
    dsr_type_to_bus: dict[str, set[str]],
    dsr_dict: dict[str, DSR],
    network: Network,
) -> None:
    for dsr_name, buses in dsr_type_to_bus.items():
        network.add_dsr(dsr_dict[dsr_name])
        for bus_name in buses:
            network.buses[bus_name].dsr_type = dsr_name


def _test_shift_variable(
    network: Network,
    indices: Indices,
    opt_config: OptConfig,
    n_buses: int,
    var_name: str,
) -> None:
    """Test if shift variables work correctly."""
    var = getattr(BusVariables(Model(), network, indices, opt_config), var_name)
    assert (
        len(var) == n_buses
    ), f"incorrect number of {var_name} variables: {len(var)}, should be {len(network.dsr)}"
    assert all(
        v.shape == (len(indices.H), len(indices.Y)) for v in var.values()
    ), f"not every variable in variable dictionary {var_name} has shape {len(indices.H), len(indices.Y)}"


def _test_ens_variables(
    network: Network, indices: Indices, opt_config: OptConfig
) -> None:
    var = BusVariables(Model(), network, indices, opt_config).bus_ens
    assert var.shape == (
        len(indices.BUS),
        len(indices.H),
        len(indices.Y),
    ), f"ens variable has incorrect shape {var.shape}, should be {len(indices.H), len(indices.Y)}"


@pytest.mark.parametrize(
    ("h_sample", "y_sample", "dsr_type_bus"),
    [
        (
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4],
            {"dsr1": {lbs_bus_name(0, HEAT), lbs_bus_name(1, HEAT)}},
        ),
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [0, 1, 2, 3],
            {
                "dsr1": {lbs_bus_name(0, HEAT), lbs_bus_name(1, HEAT)},
                "dsr2": {lbs_bus_name(0, EE)},
            },
        ),
        (
            [0, 1, 2],
            [0, 1],
            dict(),
        ),
    ],
)
def test_bus_variables(
    h_sample: np.ndarray,
    y_sample: np.ndarray,
    dsr_type_bus: dict[str, set[str]],
    dsr_factory: Callable[..., DSR],
    complete_network: Network,
    opt_config: OptConfig,
) -> None:
    opt_config.hour_sample, opt_config.year_sample = h_sample, y_sample
    indices = Indices(complete_network, opt_config)
    dsr_dict = {dsr_name: dsr_factory(dsr_name) for dsr_name in dsr_type_bus}
    n_buses = len(set().union(*dsr_type_bus.values()))
    _add_dsr_to_network(dsr_type_bus, dsr_dict, complete_network)
    _test_ens_variables(complete_network, indices, opt_config)
    _test_shift_variable(
        complete_network, indices, opt_config, n_buses, var_name="shift_plus"
    )
    _test_shift_variable(
        complete_network, indices, opt_config, n_buses, var_name="shift_minus"
    )
