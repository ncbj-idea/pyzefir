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

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network import Network
from pyzefir.model.network_elements import (
    AggregatedConsumer,
    Bus,
    CapacityFactor,
    DemandProfile,
    EmissionFee,
    Fuel,
    Generator,
    Line,
    LocalBalancingStack,
    Storage,
    StorageType,
    TransmissionFee,
)
from tests.unit.defaults import (
    CO2_EMISSION,
    ELECTRICITY,
    HEATING,
    PM10_EMISSION,
    TRANSPORT,
    default_energy_profile,
    default_network_constants,
    default_yearly_demand,
    get_default_generator_type,
    get_default_storage_type,
)


def test_network_init(network: Network) -> None:
    assert network.energy_types == [ELECTRICITY, HEATING]
    assert len(network.buses) == 0
    assert len(network.generators) == 0
    assert len(network.storages) == 0
    assert len(network.lines) == 0
    assert len(network.local_balancing_stacks) == 0
    assert len(network.aggregated_consumers) == 0


def test_add_bus(network: Network) -> None:
    bus_a = Bus(name="bus_A", energy_type=ELECTRICITY)
    bus_b = Bus(name="bus_B", energy_type=HEATING)
    network.add_bus(bus_a)
    network.add_bus(bus_b)
    assert "bus_A" in network.buses
    assert "bus_B" in network.buses
    with pytest.raises(NetworkValidatorException) as e_info:
        duplicated_bus_b = Bus(name="bus_B", energy_type=HEATING)
        network.add_bus(duplicated_bus_b)
    assert len(network.buses) == 2
    assert (
        str(e_info.value)
        == "Network element Bus with name bus_B has been already added"
    )

    bus_c = Bus(name="bus_C", energy_type=TRANSPORT)
    with pytest.raises(NetworkValidatorException) as e_info:
        network.add_bus(bus_c)
    assert len(e_info.value.exceptions) == 1
    assert str(e_info.value.exceptions[0]) == (
        "Bus bus_C has energy type TRANSPORT which "
        "is not compliant with the network energy types: ['ELECTRICITY', 'HEATING']"
    )
    bus_d = Bus(name="bus_D", energy_type=124224)  # noqa
    with pytest.raises(NetworkValidatorException) as e_info:
        network.add_bus(bus_d)
    assert len(e_info.value.exceptions) == 1
    assert (
        str(e_info.value.exceptions[0])
        == "Energy Type must be a string, but given <class 'int'> instead"
    )

    assert len(network.buses) == 2


def test_add_storage(network: Network) -> None:
    bus_a = Bus(name="bus_A", energy_type=ELECTRICITY)
    network.add_bus(bus_a)
    storage_type = get_default_storage_type(series_length=network.constants.n_years)
    network.add_storage_type(storage_type)
    storage = Storage(
        name="storage_a",
        bus="bus_A",
        energy_source_type=storage_type.name,
        unit_base_cap=20,
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
    with pytest.raises(NetworkValidatorExceptionGroup) as e_info:
        storage_b = Storage(
            name="storage_b",
            bus="bus_B",
            energy_source_type=storage_type.name,
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
        network.add_storage(storage_b)
    assert len(e_info.value.exceptions) == 1
    assert (
        str(e_info.value.exceptions[0])
        == f"Bus {storage_b.bus} does not exist in the network"
    )
    with pytest.raises(NetworkValidatorException) as e_info_duplicated:
        duplicated_storage = Storage(
            name="storage_a",
            bus="bus_A",
            energy_source_type=storage_type.name,
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
        network.add_storage(duplicated_storage)
    assert (
        str(e_info_duplicated.value)
        == "Network element Storage with name storage_a has been already added"
    )
    bus_B = Bus(name="bus_B", energy_type=HEATING)
    network.add_bus(bus_B)
    storage_c = Storage(
        name="storage_c",
        bus="bus_B",
        energy_source_type=storage_type.name,
        unit_base_cap=100,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
    )
    with pytest.raises(NetworkValidatorExceptionGroup) as e_info:
        network.add_storage(storage_c)
    assert len(e_info.value.exceptions) == 1
    assert str(e_info.value.exceptions[0]) == (
        "Bus bus_B energy type (HEATING) is different, than the storage "
        "storage_c energy type (ELECTRICITY) attached to this bus"
    )
    assert len(network.storages) == 1


@pytest.fixture
def network_gen(network: Network) -> Network:
    coal = Fuel(
        name="coal",
        cost=pd.Series([1, 2, 3, 4]),
        availability=pd.Series([1, 2, 3, 4]),
        emission={},
        energy_per_unit=0.4,
    )
    network.add_fuel(coal)
    bus_a = Bus(name="bus_A", energy_type=ELECTRICITY)
    bus_b = Bus(name="bus_B", energy_type=HEATING)
    gen_type_a = get_default_generator_type(series_length=network.constants.n_years)
    gen_a = Generator(
        name="gen_A",
        energy_source_type=gen_type_a.name,
        bus={"bus_A", "bus_B"},
        unit_base_cap=25,
        unit_min_capacity=pd.Series([np.nan] * network.constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * network.constants.n_years),
        unit_min_capacity_increase=pd.Series([np.nan] * network.constants.n_years),
        unit_max_capacity_increase=pd.Series([np.nan] * network.constants.n_years),
    )
    network.add_bus(bus_a)
    network.add_bus(bus_b)
    network.add_generator_type(gen_type_a)
    network.add_generator(gen_a)
    return network


def test_add_duplicated_generator(network_gen: Network) -> None:
    gen_type = get_default_generator_type(series_length=network_gen.constants.n_years)
    duplicated_gen_a = Generator(
        name="gen_A",
        energy_source_type=gen_type.name,
        bus={"bus_A", "bus_B"},
        unit_base_cap=0.001,
        unit_min_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * default_network_constants.n_years),
        unit_min_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
        unit_max_capacity_increase=pd.Series(
            [np.nan] * default_network_constants.n_years
        ),
    )
    with pytest.raises(NetworkValidatorException) as e_info:
        network_gen.add_generator(duplicated_gen_a)
    assert (
        str(e_info.value)
        == "Network element Generator with name gen_A has been already added"
    )


def test_add_generator_non_instance(network_gen: Network) -> None:
    gen_non_instance = "gen_non_instance"
    with pytest.raises(NetworkValidatorException) as e_info:
        network_gen.add_generator(gen_non_instance)  # noqa
    assert (
        str(e_info.value)
        == "Generator must be an instance of Generator class, but it is <class 'str'> instead."
    )


def test_add_line(network: Network) -> None:
    bus_a = Bus(name="bus_A", energy_type=ELECTRICITY)
    bus_b = Bus(name="bus_B", energy_type=HEATING)
    bus_c = Bus(name="bus_C", energy_type=ELECTRICITY)
    network.add_bus(bus_a)
    network.add_bus(bus_b)
    network.add_bus(bus_c)
    line_name = "A->C"
    line = Line(
        name=line_name,
        fr="bus_A",
        to="bus_C",
        energy_type=ELECTRICITY,
        transmission_loss=1e-3,
        max_capacity=100.0,
    )
    network.add_line(line)
    assert line_name in network.lines
    line_between_two_energy_types_name = "line_between_to_energy_types"
    line_between_two_energy_types = Line(
        name=line_between_two_energy_types_name,
        energy_type=ELECTRICITY,
        fr="bus_A",
        to="bus_B",
        max_capacity=100.0,
        transmission_loss=1e-3,
    )
    with pytest.raises(NetworkValidatorExceptionGroup) as e_info:
        network.add_line(line_between_two_energy_types)
    assert len(e_info.value.exceptions) == 2
    assert str(e_info.value.exceptions[0]) == (
        "Cannot set end of the line line_between_to_energy_types to bus bus_B. "
        "Bus bus_B energy type is HEATING, which is different from the line "
        "line_between_to_energy_types energy type: ELECTRICITY."
    )
    assert str(e_info.value.exceptions[1]) == (
        "Cannot add a line line_between_to_energy_types between buses bus_A and "
        "bus_B with different energy types ELECTRICITY != HEATING"
    )
    line = Line(
        name="None->None",
        fr="None",
        to="None2",
        energy_type=ELECTRICITY,
        transmission_loss=1e-3,
        max_capacity=100.0,
    )
    with pytest.raises(NetworkValidatorExceptionGroup) as e_info:
        network.add_line(line)
    assert len(network.lines) == 1
    assert len(e_info.value.exceptions) == 2
    assert str(e_info.value.exceptions[0]) == (
        "Cannot set the beginning of the line None->None to bus None. Bus None does not exist in the network"
    )
    assert str(e_info.value.exceptions[1]) == (
        "Cannot set the end of the line None->None to bus None2. Bus None2 does not exist in the network"
    )


def test_heating_line_between_electric_buses() -> None:
    network = Network(
        energy_types=[ELECTRICITY],
        network_constants=default_network_constants,
        emission_types=[CO2_EMISSION, PM10_EMISSION],
    )
    bus_a = Bus(name="bus_A", energy_type=ELECTRICITY)
    bus_c = Bus(name="bus_C", energy_type=ELECTRICITY)
    network.add_bus(bus_a)
    network.add_bus(bus_c)
    line_name = "A->C"
    line = Line(
        name=line_name,
        fr="bus_A",
        to="bus_C",
        energy_type=HEATING,
        transmission_loss=1e-3,
        max_capacity=100.0,
    )
    with pytest.raises(NetworkValidatorExceptionGroup) as e_info:
        network.add_line(line)
    assert len(e_info.value.exceptions) == 3
    assert str(e_info.value.exceptions[0]) == (
        "Energy type of line HEATING not found in the Network energy types: ['ELECTRICITY']"
    )
    assert str(e_info.value.exceptions[1]) == (
        "Cannot set beginning of the line A->C to bus bus_A. Bus bus_A energy "
        "type is ELECTRICITY, which is different from the line A->C "
        "energy type: HEATING."
    )
    assert str(e_info.value.exceptions[2]) == (
        "Cannot set end of the line A->C to bus bus_C. Bus bus_C energy type "
        "is ELECTRICITY, which is different from the line A->C energy type: HEATING."
    )


def test_if_network_validates_local_balancing_stack() -> None:
    network = Network(
        energy_types=[ELECTRICITY],
        network_constants=default_network_constants,
        emission_types=[CO2_EMISSION, PM10_EMISSION],
    )
    bus_a = Bus(name="bus_A", energy_type=ELECTRICITY)
    bus_b = Bus(name="bus_B", energy_type=ELECTRICITY)
    network.add_bus(bus_a)
    network.add_bus(bus_b)
    lb_stack_1 = LocalBalancingStack("1", buses_out={ELECTRICITY: "bus_A"})
    lb_stack_1.validate = MagicMock()
    network.add_local_balancing_stack(lb_stack_1)
    lb_stack_1.validate.assert_called_once_with(network)


def test_add_duplicated_local_balancing_stack() -> None:
    network = Network(
        energy_types=[ELECTRICITY],
        network_constants=default_network_constants,
        emission_types=[CO2_EMISSION, PM10_EMISSION],
    )
    bus_a = Bus(name="bus_A", energy_type=ELECTRICITY)
    bus_b = Bus(name="bus_B", energy_type=ELECTRICITY)
    network.add_bus(bus_a)
    network.add_bus(bus_b)
    lb_stack_1 = LocalBalancingStack(
        "1", buses_out={ELECTRICITY: "bus_A"}, buses={ELECTRICITY: {"bus_A", "bus_B"}}
    )

    network.add_local_balancing_stack(lb_stack_1)
    assert len(network.local_balancing_stacks) == 1

    lb_stack_2 = LocalBalancingStack(
        "1", buses_out={ELECTRICITY: "bus_B"}, buses={ELECTRICITY: {"bus_A", "bus_B"}}
    )
    with pytest.raises(NetworkValidatorException) as e_info:
        network.add_local_balancing_stack(lb_stack_2)
    assert (
        str(e_info.value)
        == "Network element LocalBalancingStack with name 1 has been already added"
    )
    assert len(network.local_balancing_stacks) == 1


def test_add_duplicated_aggregated_consumer(network: Network) -> None:
    network.add_bus(Bus(name="bus_A", energy_type=ELECTRICITY))
    network.add_bus(Bus(name="bus_B", energy_type=ELECTRICITY))
    network.add_bus(Bus(name="bus_C", energy_type=HEATING))

    network.add_local_balancing_stack(
        LocalBalancingStack(
            name="lb_A",
            buses_out={ELECTRICITY: "bus_A", HEATING: "bus_C"},
            buses={ELECTRICITY: {"bus_A", "bus_B"}, HEATING: {"bus_C"}},
        )
    )
    network.add_local_balancing_stack(
        LocalBalancingStack(
            name="lb_B",
            buses_out={ELECTRICITY: "bus_B", HEATING: "bus_C"},
            buses={ELECTRICITY: {"bus_A", "bus_B"}, HEATING: {"bus_C"}},
        )
    )
    assert len(network.aggregated_consumers) == 0
    demand_profile = DemandProfile("default", default_energy_profile())
    network.add_demand_profile(demand_profile)

    aggregate = AggregatedConsumer(
        "aggr_A",
        demand_profile.name,
        {"lb_A": 0.3, "lb_B": 0.7},
        default_yearly_demand,
        {
            "lb_A": pd.Series([np.nan] * default_network_constants.n_years),
            "lb_B": pd.Series([np.nan] * default_network_constants.n_years),
        },
        {
            "lb_A": pd.Series([np.nan] * default_network_constants.n_years),
            "lb_B": pd.Series([np.nan] * default_network_constants.n_years),
        },
        {
            "lb_A": pd.Series([np.nan] * default_network_constants.n_years),
            "lb_B": pd.Series([np.nan] * default_network_constants.n_years),
        },
        {
            "lb_A": pd.Series([np.nan] * default_network_constants.n_years),
            "lb_B": pd.Series([np.nan] * default_network_constants.n_years),
        },
        pd.Series([1000] * default_network_constants.n_years),
        None,
    )
    network.add_aggregated_consumer(aggregate)
    assert len(network.aggregated_consumers) == 1

    aggregate2 = AggregatedConsumer(
        "aggr_A",
        demand_profile.name,
        {"lb_A": 0.1, "lb_B": 0.9},
        default_yearly_demand,
        {
            "lb_A": pd.Series([np.nan] * default_network_constants.n_years),
            "lb_B": pd.Series([np.nan] * default_network_constants.n_years),
        },
        {
            "lb_A": pd.Series([np.nan] * default_network_constants.n_years),
            "lb_B": pd.Series([np.nan] * default_network_constants.n_years),
        },
        {
            "lb_A": pd.Series([np.nan] * default_network_constants.n_years),
            "lb_B": pd.Series([np.nan] * default_network_constants.n_years),
        },
        {
            "lb_A": pd.Series([np.nan] * default_network_constants.n_years),
            "lb_B": pd.Series([np.nan] * default_network_constants.n_years),
        },
        pd.Series([1000] * default_network_constants.n_years),
        None,
    )
    with pytest.raises(NetworkValidatorException) as e_info:
        network.add_aggregated_consumer(aggregate2)
    assert (
        str(e_info.value)
        == "Network element AggregatedConsumer with name aggr_A has been already added"
    )
    assert len(network.aggregated_consumers) == 1


@pytest.mark.parametrize(
    "fuel, expected_info",
    (
        (None, "Fuel cannot be None"),
        (
            Fuel("test", None, None, None, None),  # noqa
            (
                "Availability must be a pandas Series, but NoneType given",
                "Cost must be a pandas Series, but NoneType given",
                "Energy per unit must be of float type",
                "Emission mapping must be of dict type",
            ),
        ),
        (
            Fuel(
                "test",
                None,  # noqa
                pd.Series([1] * default_network_constants.n_years),
                pd.Series([1] * default_network_constants.n_years),
                0.1,
            ),
            "Emission mapping must be of dict type",
        ),
        (
            Fuel("test", dict(CO2_EMISSION=0.1), None, pd.Series(), 0.1),  # noqa
            "Availability must be a pandas Series, but NoneType given",
        ),
        (
            Fuel(
                "test",
                dict(CO2_EMISSION=0.1),
                pd.Series([1] * default_network_constants.n_years),
                None,  # noqa
                0.1,
            ),
            "Cost must be a pandas Series, but NoneType given",
        ),
        (
            Fuel(
                "test",
                dict(CO2_EMISSION=0.1),
                pd.Series([1] * default_network_constants.n_years),
                pd.Series([1] * default_network_constants.n_years),
                [124],  # noqa
            ),
            "Energy per unit must be of float type",
        ),
        (
            Fuel(
                "test",
                dict(P12_EMISSION=0.1),
                pd.Series([1] * default_network_constants.n_years),
                pd.Series([1] * default_network_constants.n_years),
                0.1,
            ),
            "Emission type P12_EMISSION not found in network",
        ),
    ),
)
def test_add_incorrect_fuel(fuel: Fuel, expected_info: str | tuple[str]) -> None:
    network = Network(
        energy_types=[HEATING],
        emission_types=[CO2_EMISSION],
        network_constants=default_network_constants,
    )
    with pytest.raises(NetworkValidatorException) as e_info:
        network.add_fuel(fuel)
    if e_info.type is NetworkValidatorExceptionGroup:
        if isinstance(expected_info, tuple):
            expected_exceptions = {str(e) for e in e_info.value.exceptions}
            actual_exceptions = set(expected_info)
            assert actual_exceptions == expected_exceptions
        else:
            assert str(e_info.value.exceptions[0]) == expected_info
    else:
        assert str(e_info.value) == expected_info


def test_add_fuel() -> None:
    network = Network(
        energy_types=[HEATING],
        network_constants=default_network_constants,
    )
    assert len(network.fuels) == 0
    network.add_fuel(
        Fuel(
            name="lorem",
            cost=pd.Series([1, 2, 3, 4]),
            availability=pd.Series([1, 2, 3, 4]),
            emission={},
            energy_per_unit=5.0,
        )
    )
    assert len(network.fuels) == 1
    network.add_fuel(
        Fuel(
            name="ipsum",
            cost=pd.Series([1, 2, 3, 4]),
            availability=pd.Series([1, 2, 3, 4]),
            emission={},
            energy_per_unit=5.0,
        )
    )
    assert len(network.fuels) == 2


def test_add_capacity_factor() -> None:
    network = Network(
        energy_types=[HEATING],
        network_constants=default_network_constants,
        emission_types=[CO2_EMISSION, PM10_EMISSION],
    )
    assert len(network.capacity_factors) == 0
    network.add_capacity_factor(
        CapacityFactor(name="sun", profile=pd.Series(data=np.arange(24)))
    )
    assert len(network.capacity_factors) == 1
    network.add_capacity_factor(
        CapacityFactor(name="wukong", profile=pd.Series(data=np.arange(24)))
    )
    assert len(network.capacity_factors) == 2


def test_add_duplicated_fuel() -> None:
    network = Network(
        energy_types=[HEATING],
        network_constants=default_network_constants,
        emission_types=[CO2_EMISSION, PM10_EMISSION],
    )
    assert len(network.fuels) == 0
    network.add_fuel(
        Fuel(
            name="lorem",
            cost=pd.Series([1, 2, 3, 4]),
            availability=pd.Series([1, 2, 3, 4]),
            emission={},
            energy_per_unit=0.4,
        )
    )
    assert len(network.fuels) == 1
    with pytest.raises(NetworkValidatorException) as e_info:
        network.add_fuel(
            Fuel(
                name="lorem",
                cost=pd.Series([1, 2, 3, 4]),
                availability=pd.Series([1, 2, 3, 4]),
                emission={},
                energy_per_unit=0.4,
            )
        )
    assert (
        str(e_info.value)
        == "Network element Fuel with name lorem has been already added"
    )
    assert len(network.fuels) == 1


def test_add_transmission_fee() -> None:
    network = Network(
        energy_types=[HEATING],
        network_constants=default_network_constants,
        emission_types=[CO2_EMISSION, PM10_EMISSION],
    )
    assert len(network.transmission_fees) == 0

    network.add_transmission_fee(
        TransmissionFee(name="Test Fee", fee=pd.Series(data=np.arange(24)))
    )

    assert len(network.transmission_fees) == 1

    with pytest.raises(NetworkValidatorException) as e_info:
        network.add_transmission_fee(
            TransmissionFee(name="Test Fee", fee=pd.Series(data=np.arange(24)))
        )
    assert (
        str(e_info.value)
        == "Network element TransmissionFee with name Test Fee has been already added"
    )
    assert len(network.transmission_fees) == 1


def test_add_emission_fee() -> None:
    network = Network(
        energy_types=[HEATING],
        network_constants=default_network_constants,
        emission_types=[CO2_EMISSION],
    )
    assert len(network.emission_fees) == 0

    network.add_emission_fee(
        EmissionFee(
            name="Emission_Fee_Test",
            emission_type=CO2_EMISSION,
            price=pd.Series([0.1] * network.constants.n_years),
        )
    )

    assert len(network.emission_fees) == 1

    with pytest.raises(NetworkValidatorException) as e_info:
        network.add_emission_fee(
            EmissionFee(
                name="Emission_Fee_Test",
                emission_type=CO2_EMISSION,
                price=pd.Series([0.1] * network.constants.n_years),
            )
        )
    assert (
        str(e_info.value)
        == "Network element EmissionFee with name Emission_Fee_Test has been already added"
    )
    assert len(network.emission_fees) == 1


def test_add_energy_loss() -> None:
    network = Network(
        energy_types=[HEATING],
        network_constants=default_network_constants,
        emission_types=[CO2_EMISSION],
    )
    network.add_storage_type(
        stor_type=StorageType(
            name="TestStorType",
            energy_type=HEATING,
            build_time=0,
            life_time=5,
            capex=pd.Series([10] * 4),
            opex=pd.Series([0] * 4),
            generation_efficiency=1.0,
            load_efficiency=1.0,
            power_to_capacity=1.0,
            power_utilization=1.0,
            min_capacity=pd.Series([np.nan] * 4),
            max_capacity=pd.Series([np.nan] * 4),
            min_capacity_increase=pd.Series([np.nan] * 4),
            max_capacity_increase=pd.Series([np.nan] * 4),
            cycle_length=100,
            energy_loss=0.1,
        )
    )
    assert network.storage_types["TestStorType"].energy_loss == 0.1
    assert len(network.storage_types) == 1
