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

import sys

import numpy as np
import pandas as pd
import pytest

from pyzefir.graph.utils import draw_network
from pyzefir.model.network import Network
from pyzefir.model.network_elements import (
    AggregatedConsumer,
    Bus,
    DemandProfile,
    Fuel,
    Generator,
    Line,
    LocalBalancingStack,
    Storage,
)
from tests.unit.defaults import (
    ELECTRICITY,
    HEATING,
    TRANSPORT,
    default_network_constants,
    get_default_generator_type,
    get_default_storage_type,
)

try:
    import pygraphviz  # noqa: F401 'pygraphviz' imported but unused
except ImportError:
    pass


def simple_network() -> Network:
    network = Network(
        energy_types=[ELECTRICITY, HEATING, TRANSPORT],
        network_constants=default_network_constants,
    )
    bus_a = Bus(name="bus_A", energy_type=ELECTRICITY)
    bus_b = Bus(name="bus_B", energy_type=HEATING)
    bus_c = Bus(name="bus_C", energy_type=ELECTRICITY)
    bus_d = Bus(name="bus_D", energy_type=TRANSPORT)
    coal = Fuel(
        name="coal",
        cost=pd.Series([1, 2, 3, 4]),
        availability=pd.Series([1, 2, 3, 4]),
        emission={},
        energy_per_unit=3.5,
    )
    network.add_fuel(coal)
    network.add_bus(bus_a)
    network.add_bus(bus_b)
    network.add_bus(bus_c)
    network.add_bus(bus_d)
    storage_b_energy_type = get_default_storage_type(
        name="storage_b_energy_type",
        energy_type=HEATING,
        series_length=network.constants.n_years,
    )
    storage_a_energy_type = get_default_storage_type(
        name="storage_a_energy_type", series_length=network.constants.n_years
    )
    generator_a_energy_type = get_default_generator_type(
        name="generator_a_energy_type",
        energy_types={ELECTRICITY, HEATING, TRANSPORT},
        emission_reduction=dict(),
        series_length=network.constants.n_years,
    )
    generator_c_energy_type = get_default_generator_type(
        name="generator_c_energy_type",
        emission_reduction=dict(),
        series_length=network.constants.n_years,
    )
    network.add_storage_type(storage_a_energy_type)
    network.add_storage_type(storage_b_energy_type)
    network.add_generator_type(generator_a_energy_type)
    network.add_generator_type(generator_c_energy_type)
    storage = Storage(
        name="storage_b",
        bus="bus_B",
        energy_source_type=storage_b_energy_type.name,
        unit_base_cap=10,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
    )
    network.add_storage(storage)
    storage = Storage(
        name="storage_a",
        bus="bus_A",
        energy_source_type=storage_a_energy_type.name,
        unit_base_cap=15,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
    )
    network.add_storage(storage)
    gen_a = Generator(
        name="gen_A",
        energy_source_type=generator_a_energy_type.name,
        bus={"bus_A", "bus_B"},
        unit_base_cap=40,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
    )
    gen_c = Generator(
        name="gen_C",
        energy_source_type=generator_c_energy_type.name,
        bus={"bus_B", "bus_C"},
        unit_base_cap=40,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
    )
    network.add_generator(gen_a)
    network.add_generator(gen_c)
    line_name = "A->C"
    line = Line(
        name=line_name,
        fr="bus_A",
        to="bus_C",
        transmission_loss=1e-3,
        max_capacity=100.0,
        energy_type=ELECTRICITY,
    )
    network.add_line(line)
    lb_stack = LocalBalancingStack(
        name="lb_stack_CBD",
        buses_out={HEATING: "bus_B", ELECTRICITY: "bus_C", TRANSPORT: "bus_D"},
    )
    network.add_local_balancing_stack(lb_stack)
    lb_stack_2 = LocalBalancingStack(
        name="lb_stack_ABD",
        buses_out={HEATING: "bus_B", ELECTRICITY: "bus_A", TRANSPORT: "bus_D"},
    )
    network.add_local_balancing_stack(lb_stack_2)
    demand_profile = DemandProfile(
        "default",
        {
            ELECTRICITY: pd.Series(data=[0, 1]),
            HEATING: pd.Series(data=[1, 0]),
            TRANSPORT: pd.Series(data=[0.5, 0.5]),
        },
    )
    network.add_demand_profile(demand_profile)
    aggregate = AggregatedConsumer(
        "aggr_A",
        demand_profile.name,
        {"lb_stack_CBD": 0.3, "lb_stack_ABD": 0.7},
        {ELECTRICITY: pd.Series(), HEATING: pd.Series(), TRANSPORT: pd.Series()},
        {
            "lb_stack_CBD": pd.Series([np.nan] * default_network_constants.n_years),
            "lb_stack_ABD": pd.Series([np.nan] * default_network_constants.n_years),
        },
        {
            "lb_stack_CBD": pd.Series([np.nan] * default_network_constants.n_years),
            "lb_stack_ABD": pd.Series([np.nan] * default_network_constants.n_years),
        },
        {
            "lb_stack_CBD": pd.Series([np.nan] * default_network_constants.n_years),
            "lb_stack_ABD": pd.Series([np.nan] * default_network_constants.n_years),
        },
        {
            "lb_stack_CBD": pd.Series([np.nan] * default_network_constants.n_years),
            "lb_stack_ABD": pd.Series([np.nan] * default_network_constants.n_years),
        },
        n_consumers=pd.Series([1000] * default_network_constants.n_years),
        average_area=None,
    )
    network.add_aggregated_consumer(aggregate)
    return network


@pytest.mark.skipif(
    "pygraphviz" not in sys.modules,
    reason="Requires the PyGraphviz library (currently not available on CICD)",
)
def test_draw_network() -> None:
    network = simple_network()
    draw_network(network, show=False)
