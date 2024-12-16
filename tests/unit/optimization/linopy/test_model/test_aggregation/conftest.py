import pandas as pd
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
    StorageType,
    TransmissionFee,
)
from pyzefir.model.utils import NetworkConstants
from tests.unit.optimization.linopy.names import CO2, EE, HEAT, PM10


@pytest.fixture
def network_without_grid(
    network_constants: NetworkConstants,
    fuels: dict[str, Fuel],
    cfs: dict[str, CapacityFactor],
    demand_profile: DemandProfile,
    generator_types: dict[str, GeneratorType],
    storage_types: dict[str, StorageType],
    grid_bus: Bus,
    hs_bus: Bus,
    local_ee_bus: Bus,
    local_ee_bus2: Bus,
    local_heat_bus: Bus,
    local_heat_bus2: Bus,
    heating_system_connection: Line,
    heating_system_connection2: Line,
    coal_power_plant: Generator,
    local_coal_heat_plant: Generator,
    local_coal_heat_plant2: Generator,
    lbs: LocalBalancingStack,
    aggr: AggregatedConsumer,
) -> Network:
    """
    Network used all tests in this module contains
        * two lbs, two local heat plan systems
        * one aggregate consumer
    """

    result = Network(
        energy_types=[HEAT, EE],
        emission_types=[CO2, PM10],
        network_constants=network_constants,
    )
    result.add_fuel(fuels["coal"])
    result.add_fuel(fuels["biomass"])
    pp_coal = generator_types["pp_coal"]
    pp_coal.generation_compensation = pd.Series([2, 2, 2, 2, 2])
    result.add_generator_type(generator_types["pp_coal"])
    result.add_generator_type(generator_types["local_coal_heat_plant"])
    result.add_generator_type(generator_types["local_coal_heat_plant2"])
    result.add_demand_profile(demand_profile)

    result.add_bus(local_ee_bus)
    result.add_bus(local_ee_bus2)
    result.add_bus(grid_bus)
    result.add_generator(coal_power_plant)

    result.add_bus(local_heat_bus)
    result.add_bus(local_heat_bus2)
    result.add_bus(hs_bus)
    result.add_line(heating_system_connection)
    result.add_line(heating_system_connection2)
    result.add_generator(local_coal_heat_plant)
    result.add_generator(local_coal_heat_plant2)

    result.add_local_balancing_stack(lbs)
    result.add_aggregated_consumer(aggr)

    return result


@pytest.fixture
def network(
    network_without_grid: Network,
    transmission_fee: TransmissionFee,
    grid_connection: Line,
    grid_connection2: Line,
) -> Network:
    """
    Network used all tests in this module contains
        * two lbs, two local heat plan systems
        * one aggregate consumer
        * grid bus
        * grid line
    """

    result = network_without_grid
    result.add_transmission_fee(transmission_fee)
    result.add_line(grid_connection)
    result.add_line(grid_connection2)

    return result
