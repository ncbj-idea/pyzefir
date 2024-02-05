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
    DemandProfile,
    Fuel,
    Generator,
    GeneratorType,
    Line,
    LocalBalancingStack,
    TransmissionFee,
)
from pyzefir.model.network_elements.emission_fee import EmissionFee
from pyzefir.model.utils import NetworkConstants
from tests.unit.optimization.gurobi.names import CO2, EE, HEAT, PM10


@pytest.fixture
def network(
    network_constants: NetworkConstants,
    fuels: dict[str, Fuel],
    demand_profile: DemandProfile,
    generator_types: dict[str, GeneratorType],
    grid_bus: Bus,
    hs_bus: Bus,
    local_ee_bus: Bus,
    local_heat_bus: Bus,
    heating_system_connection: Line,
    transmission_fee: TransmissionFee,
    emission_fee_CO2: EmissionFee,
    emission_fee_PM10: EmissionFee,
    grid_connection: Line,
    coal_power_plant: Generator,
    biomass_heat_plant: Generator,
    coal_heat_plant: Generator,
    coal_chp: Generator,
    lbs: LocalBalancingStack,
    aggr: AggregatedConsumer,
) -> Network:
    """
    Network used all tests in this module contains
        * two global buses:
            - grid_bus with coal power plant and chp coal
            - heating_system_bus with biomass heat coal heat plant and chp coal
        * one local balancing stack (lbs) connected to grid_bus and heating_system bus (no local energy sources)
        * one aggregated_consumer (aggr) connected to the local balancing stack (lbs)
        * lines losses = 0
        * lines capacity = inf

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
    result.add_generator_type(generator_types["heat_plant_coal"])
    result.add_generator_type(generator_types["chp_coal"])
    result.add_demand_profile(demand_profile)
    result.add_emission_fee(emission_fee_CO2)
    result.add_emission_fee(emission_fee_PM10)

    result.add_bus(local_ee_bus)
    result.add_bus(grid_bus)
    result.add_transmission_fee(transmission_fee)
    result.add_line(grid_connection)
    result.add_generator(coal_power_plant)

    result.add_bus(local_heat_bus)
    result.add_bus(hs_bus)
    result.add_line(heating_system_connection)
    result.add_generator(biomass_heat_plant)
    result.add_generator(coal_heat_plant)
    result.add_generator(coal_chp)

    result.add_local_balancing_stack(lbs)
    result.add_aggregated_consumer(aggr)

    return result
