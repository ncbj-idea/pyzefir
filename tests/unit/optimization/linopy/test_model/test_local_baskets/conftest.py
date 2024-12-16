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
from tests.unit.optimization.linopy.names import CO2, EE, PM10


@pytest.fixture
def network(
    network_constants: NetworkConstants,
    fuels: dict[str, Fuel],
    cfs: dict[str, CapacityFactor],
    demand_profile_ee: DemandProfile,
    generator_types: dict[str, GeneratorType],
    storage_types: dict[str, StorageType],
    grid_bus: Bus,
    local_ee_bus: Bus,
    local_ee_bus2: Bus,
    transmission_fee: TransmissionFee,
    grid_connection: Line,
    grid_connection2: Line,
    coal_power_plant: Generator,
    local_pv: Generator,
    local_pv2: Generator,
    lbs_ee: LocalBalancingStack,
    lbs_ee2: LocalBalancingStack,
    aggr_ee: AggregatedConsumer,
) -> Network:
    """
    Network used all tests in this module contains
        * two lbs, two local heat plan systems
        * one global grid
        * one aggregate consumer
    """

    result = Network(
        energy_types=[EE],
        emission_types=[CO2, PM10],
        network_constants=network_constants,
    )

    result.add_fuel(fuels["coal"])
    result.add_generator_type(generator_types["pp_coal"])
    result.add_demand_profile(demand_profile_ee)

    result.add_bus(local_ee_bus)
    result.add_bus(local_ee_bus2)
    result.add_bus(grid_bus)
    result.add_transmission_fee(transmission_fee)
    result.add_line(grid_connection2)
    result.add_generator(coal_power_plant)

    result.add_local_balancing_stack(lbs_ee)
    result.add_local_balancing_stack(lbs_ee2)
    result.add_aggregated_consumer(aggr_ee)

    result.add_capacity_factor(cfs["sun"])
    result.add_generator_type(generator_types["pv"])
    result.add_generator(local_pv)
    result.add_generator(local_pv2)

    return result
