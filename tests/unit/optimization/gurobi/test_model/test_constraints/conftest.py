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
from pyzefir.model.network_elements import (
    AggregatedConsumer,
    Bus,
    CapacityFactor,
    DemandProfile,
    Fuel,
    Generator,
    GeneratorType,
    Line,
    LocalBalancingStack,
    TransmissionFee,
)
from pyzefir.model.utils import NetworkConstants
from tests.unit.optimization.gurobi.names import CO2, EE, HEAT, PM10


@pytest.fixture
def network(
    network_constants: NetworkConstants,
    fuels: dict[str, Fuel],
    cfs: dict[str, CapacityFactor],
    demand_profile: DemandProfile,
    generator_types: dict[str, GeneratorType],
    biomass_heat_plant: Generator,
    global_solar: Generator,
    coal_power_plant: Generator,
    local_pv: Generator,
    grid_bus: Bus,
    hs_bus: Bus,
    local_ee_bus: Bus,
    local_heat_bus: Bus,
    heating_system_connection: Line,
    transmission_fee: TransmissionFee,
    grid_connection: Line,
    lbs: LocalBalancingStack,
    aggr: AggregatedConsumer,
) -> Network:
    """
    Network used all tests in this module contains
        * two global buses:
            - grid_bus with coal_power_plant
            - heating_system_bus with biomass_heat_plant + solar collectors
        * one local balancing stack (lbs) connected to grid_bus and heating_system bus (with local_pv)
        * one aggregated_consumer (aggr) connected to the local balancing stack (lbs)
        * lines losses = 0
        * lines capacity = inf

    NOTE: In particular tests, some parameters can be changed (but the network structure stays the same)
    """

    result = Network(
        energy_types=[EE, HEAT],
        emission_types=[CO2, PM10],
        network_constants=network_constants,
    )

    result.add_fuel(fuels["coal"])
    result.add_fuel(fuels["biomass"])
    result.add_capacity_factor(cfs["sun"])

    result.add_generator_type(generator_types["pv"])
    result.add_generator_type(generator_types["pp_coal"])
    result.add_generator_type(generator_types["solar"])
    result.add_generator_type(generator_types["heat_plant_biomass"])

    result.add_demand_profile(demand_profile)

    result.add_bus(grid_bus)
    result.add_bus(local_ee_bus)
    result.add_transmission_fee(transmission_fee)
    result.add_line(grid_connection)

    result.add_generator(coal_power_plant)
    result.add_generator(local_pv)

    result.add_bus(hs_bus)
    result.add_bus(local_heat_bus)
    result.add_line(heating_system_connection)

    result.add_generator(biomass_heat_plant)
    result.add_generator(global_solar)

    result.add_local_balancing_stack(lbs)
    result.add_aggregated_consumer(aggr)

    return result
