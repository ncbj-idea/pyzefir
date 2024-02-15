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
from pyzefir.model.network_elements import (
    AggregatedConsumer,
    Bus,
    DemandProfile,
    Fuel,
    Generator,
    GeneratorType,
    Line,
    LocalBalancingStack,
    TransmissionFee,
)
from pyzefir.model.utils import NetworkConstants
from tests.unit.optimization.linopy.constants import N_YEARS
from tests.unit.optimization.linopy.names import CO2, EE, HEAT, PM10


@pytest.fixture
def local_heat_pump(local_heat_bus: Bus, local_ee_bus: Bus) -> Generator:
    """
    Heat pump generator connected to the local_heat_bus
    """
    return Generator(
        name="heat_pump_gen",
        energy_source_type="heat_pump",
        bus={local_heat_bus.name, local_ee_bus.name},
        unit_base_cap=30,
        unit_min_capacity=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity=pd.Series([np.nan] * N_YEARS),
        unit_min_capacity_increase=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity_increase=pd.Series([np.nan] * N_YEARS),
    )


@pytest.fixture
def network(
    network_constants: NetworkConstants,
    fuels: dict[str, Fuel],
    demand_profile: DemandProfile,
    generator_types: dict[str, GeneratorType],
    grid_bus: Bus,
    local_ee_bus: Bus,
    local_heat_bus: Bus,
    transmission_fee: TransmissionFee,
    grid_connection: Line,
    coal_power_plant: Generator,
    local_heat_pump: Generator,
    lbs: LocalBalancingStack,
    aggr: AggregatedConsumer,
) -> Network:
    """
    Network used all tests in this module contains
        * one global bus:
            - grid_bus with coal power plant attached to it
        * two local buses:
            - local_heat_bus with heat pump connected to it
            - local_ee_bus with grid and heat_pump connected to it
        * one local balancing stack (lbs) containing local_heat_bus and local_ee_bus
        * one aggregated_consumer (aggr) connected to the local balancing stack (lbs)
        * lines losses = 0
        * lines capacity = inf

    NOTE: In particular tests, some parameters can be changed (but the network structure stays the same)
    """

    result = Network(
        energy_types=[HEAT, EE],
        emission_types=[CO2, PM10],
        network_constants=network_constants,
    )

    result.add_fuel(fuels["coal"])
    result.add_generator_type(generator_types["pp_coal"])
    result.add_demand_profile(demand_profile)

    result.add_bus(local_ee_bus)
    result.add_bus(grid_bus)
    result.add_transmission_fee(transmission_fee)
    result.add_line(grid_connection)
    result.add_generator(coal_power_plant)

    result.add_bus(local_heat_bus)
    result.add_generator_type(generator_types["heat_pump"])
    result.add_generator(local_heat_pump)

    result.add_local_balancing_stack(lbs)
    result.add_aggregated_consumer(aggr)

    return result
