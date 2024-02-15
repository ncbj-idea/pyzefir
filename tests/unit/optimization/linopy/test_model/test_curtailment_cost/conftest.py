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
)
from pyzefir.model.utils import NetworkConstants
from tests.unit.optimization.linopy.names import CO2, EE, HEAT, PM10


@pytest.fixture
def network(
    network_constants: NetworkConstants,
    fuels: dict[str, Fuel],
    cfs: dict[str, CapacityFactor],
    demand_profile: DemandProfile,
    generator_types: dict[str, GeneratorType],
    grid_bus: Bus,
    hs_bus: Bus,
    local_ee_bus: Bus,
    local_heat_bus: Bus,
    heating_system_connection: Line,
    grid_connection_no_fee: Line,
    coal_power_plant: Generator,
    biomass_heat_plant: Generator,
    lbs: LocalBalancingStack,
    aggr: AggregatedConsumer,
    local_pv: Generator,
) -> Network:
    """
    Network used all tests in this module contains
        * two global buses:
            - grid_bus with coal power plant attached to it
            - heating_system_bus with biomass heat plant attached to it
        * one local balancing stack (lbs) connected to grid_bus and heating_system bus (no local energy sources)
        * one aggregated_consumer (aggr) connected to the local balancing stack (lbs)
        * lines losses = 0
        * lines capacity = inf
        * added local pv
    """

    result = Network(
        energy_types=[HEAT, EE],
        emission_types=[CO2, PM10],
        network_constants=network_constants,
    )

    result.add_fuel(fuels["coal"])
    result.add_fuel(fuels["biomass"])
    result.add_generator_type(generator_types["pp_coal"])
    result.add_generator_type(generator_types["heat_plant_biomass"])
    result.add_demand_profile(demand_profile)

    result.add_bus(local_ee_bus)
    result.add_bus(grid_bus)
    # result.add_transmission_fee(transmission_fee)
    result.add_line(grid_connection_no_fee)
    result.add_generator(coal_power_plant)

    result.add_bus(local_heat_bus)
    result.add_bus(hs_bus)
    result.add_line(heating_system_connection)
    result.add_generator(biomass_heat_plant)

    result.add_local_balancing_stack(lbs)
    result.add_aggregated_consumer(aggr)

    result.add_capacity_factor(cfs["sun"])
    result.add_generator_type(generator_types["pv"])
    result.add_generator(local_pv)

    return result
