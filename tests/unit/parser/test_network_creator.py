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

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.exceptions import NetworkValidatorException
from pyzefir.model.network import Network, NetworkElementsDict
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
    Storage,
    StorageType,
    TransmissionFee,
)
from pyzefir.model.network_elements.capacity_bound import CapacityBound
from pyzefir.model.network_elements.dsr import DSR
from pyzefir.model.network_elements.emission_fee import EmissionFee
from pyzefir.model.utils import NetworkConstants
from pyzefir.parser.csv_parser import CsvParser
from pyzefir.parser.network_creator import NetworkCreator
from pyzefir.utils.path_manager import CsvPathManager
from tests.unit.defaults import (
    CO2_EMISSION,
    ELECTRICITY,
    HEATING,
    PM10_EMISSION,
    default_energy_profile,
    default_yearly_demand,
    get_default_generator_type,
    get_default_storage_type,
)


@pytest.fixture
def df_dict(csv_root_path: Path) -> dict[str, dict[str, pd.DataFrame]]:
    return CsvParser(
        path_manager=CsvPathManager(csv_root_path, scenario_name="scenario_1")
    ).load_dfs()


def test_network_creator_create_elements(
    df_dict: dict[str, dict[str, pd.DataFrame]],
) -> None:
    constants = NetworkCreator._create_network_constants(df_dict)
    buses = NetworkCreator._create_buses(df_dict)
    lines = NetworkCreator._create_lines(df_dict)
    local_stack = NetworkCreator._create_local_balancing_stacks(df_dict)
    aggregated_consumer = NetworkCreator._create_aggregated_consumers(df_dict)
    demand_profiles = NetworkCreator._create_demand_profiles(df_dict)
    generators, storages = NetworkCreator._create_energy_source_units(
        df_dict, constants
    )
    generator_types, storage_types = NetworkCreator._create_energy_source_types(
        df_dict, constants
    )

    for key, value in {
        Bus: buses,
        Line: lines,
        LocalBalancingStack: local_stack,
        AggregatedConsumer: aggregated_consumer,
        DemandProfile: demand_profiles,
        Generator: generators,
        Storage: storages,
        GeneratorType: generator_types,
        StorageType: storage_types,
    }.items():
        assert isinstance(value, tuple)
        assert all(isinstance(obj, key) for obj in value)
    assert isinstance(constants, NetworkConstants)


def test_network_creator_create_energy_emission_types(
    df_dict: dict[str, dict[str, pd.DataFrame]],
) -> None:
    types_energy = NetworkCreator._create_energy_types(df_dict)
    types_emission = NetworkCreator._create_emission_types(df_dict)

    assert isinstance(types_energy, list)
    assert isinstance(types_emission, list)
    assert all(isinstance(en_type, str) for en_type in types_energy)
    assert all(isinstance(em_type, str) for em_type in types_emission)


def test_network_creator_create_network() -> None:
    network_constants = NetworkConstants(
        4,
        24,
        {
            CO2_EMISSION: pd.Series([np.nan] * 4),
            PM10_EMISSION: pd.Series([np.nan] * 4),
        },
        {
            CO2_EMISSION: np.nan,
            PM10_EMISSION: np.nan,
        },
        {},
    )

    bus_a = Bus(
        name="bus_A",
        energy_type=ELECTRICITY,
    )
    bus_b = Bus(
        name="bus_B",
        energy_type=HEATING,
    )
    bus_c = Bus(name="bus_C", energy_type=ELECTRICITY)
    gen_type = get_default_generator_type(series_length=network_constants.n_years)
    storage_type = get_default_storage_type(series_length=network_constants.n_years)
    gen_a = Generator(
        name="gen_A",
        energy_source_type=gen_type.name,
        bus={"bus_A", "bus_B"},
        unit_base_cap=20,
        unit_min_capacity=pd.Series([np.nan] * network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * network_constants.n_years),
        unit_min_capacity_increase=pd.Series([np.nan] * network_constants.n_years),
        unit_max_capacity_increase=pd.Series([np.nan] * network_constants.n_years),
    )
    gen_b = Generator(
        name="gen_B",
        energy_source_type=gen_type.name,
        bus={"bus_A", "bus_B"},
        unit_base_cap=30,
        unit_min_capacity=pd.Series([np.nan] * network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * network_constants.n_years),
        unit_min_capacity_increase=pd.Series([np.nan] * network_constants.n_years),
        unit_max_capacity_increase=pd.Series([np.nan] * network_constants.n_years),
    )
    storage = Storage(
        name="storage_A",
        bus="bus_A",
        energy_source_type=storage_type.name,
        unit_base_cap=15,
        unit_min_capacity=pd.Series([np.nan] * network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * network_constants.n_years),
        unit_min_capacity_increase=pd.Series([np.nan] * network_constants.n_years),
        unit_max_capacity_increase=pd.Series([np.nan] * network_constants.n_years),
    )
    transmission_fee_A = TransmissionFee(
        name="fee_A", fee=pd.Series(data=np.arange(24))
    )
    transmission_fee_B = TransmissionFee(
        name="fee_B", fee=pd.Series(data=np.arange(24))
    )
    line_ac = Line(
        name="A->C",
        fr="bus_A",
        to="bus_C",
        transmission_loss=1e-3,
        max_capacity=100.0,
        energy_type=ELECTRICITY,
    )
    lb_stack_cb = LocalBalancingStack(
        name="lb_stack_CB",
        buses_out={HEATING: "bus_B", ELECTRICITY: "bus_C"},
        buses={ELECTRICITY: {"bus_A", "bus_C"}, HEATING: {"bus_B"}},
    )
    lb_stack_ab = LocalBalancingStack(
        name="lb_stack_AB",
        buses_out={HEATING: "bus_B", ELECTRICITY: "bus_A"},
        buses={ELECTRICITY: {"bus_A", "bus_C"}, HEATING: {"bus_B"}},
    )
    aggregate = AggregatedConsumer(
        "aggr_A",
        "default",
        {"lb_stack_AB": 0.3, "lb_stack_CB": 0.7},
        default_yearly_demand,
        {
            "lb_stack_AB": pd.Series([np.nan] * 4),
            "lb_stack_CB": pd.Series([np.nan] * 4),
        },
        {
            "lb_stack_AB": pd.Series([np.nan] * 4),
            "lb_stack_CB": pd.Series([np.nan] * 4),
        },
        {
            "lb_stack_AB": pd.Series([np.nan] * 4),
            "lb_stack_CB": pd.Series([np.nan] * 4),
        },
        {
            "lb_stack_AB": pd.Series([np.nan] * 4),
            "lb_stack_CB": pd.Series([np.nan] * 4),
        },
        pd.Series([10000] * 4),
        None,
    )
    coal = Fuel(
        name="coal",
        cost=pd.Series([1] * network_constants.n_years),
        availability=pd.Series([1] * network_constants.n_years),
        emission={},
        energy_per_unit=0.4,
    )
    gas = Fuel(
        name="gas",
        cost=pd.Series([1] * network_constants.n_years),
        availability=pd.Series([1] * network_constants.n_years),
        emission={},
        energy_per_unit=0.4,
    )
    demand_profile = DemandProfile("default", default_energy_profile())
    emission_fee_A = EmissionFee(
        name="em_fee_A",
        price=pd.Series([0.1] * network_constants.n_years),
        emission_type=CO2_EMISSION,
    )
    emission_fee_B = EmissionFee(
        name="em_fee_b",
        price=pd.Series([0.2] * network_constants.n_years),
        emission_type=PM10_EMISSION,
    )
    dsr = DSR(
        name="DSR_1",
        compensation_factor=0.1,
        balancing_period_len=20,
        penalization=0.1,
        relative_shift_limit=0.1,
        abs_shift_limit=None,
    )
    capacity_bound = CapacityBound(
        name="Capacity_Bound_gen_A__gen_B",
        left_technology="gen_A",
        right_technology="gen_B",
        sense="EQ",
        left_coefficient=0.4,
    )
    network = NetworkCreator._create_network(
        energy_types=[ELECTRICITY, HEATING],
        buses=(bus_a, bus_b, bus_c),
        generators=(gen_a, gen_b),
        storages=(storage,),
        lines=(line_ac,),
        local_balancing_stacks=(lb_stack_ab, lb_stack_cb),
        aggregated_consumers=(aggregate,),
        fuels=(coal, gas),
        capacity_factors=(),
        emission_types=[CO2_EMISSION, PM10_EMISSION],
        generator_types=(gen_type,),
        storage_types=(storage_type,),
        demand_profiles=(demand_profile,),
        network_constants=network_constants,
        transmission_fees=(transmission_fee_A, transmission_fee_B),
        emission_fees=(emission_fee_A, emission_fee_B),
        demand_chunks=(),
        dsr=(dsr,),
        capacity_bounds=(capacity_bound,),
        generation_fractions=(),
    )

    assert isinstance(network, Network)
    assert isinstance(network.buses, NetworkElementsDict)
    assert len(network.buses) == 3
    assert isinstance(network.generators, NetworkElementsDict)
    assert len(network.generators) == 2
    assert isinstance(network.storages, NetworkElementsDict)
    assert len(network.storages) == 1
    assert isinstance(network.lines, NetworkElementsDict)
    assert len(network.lines) == 1
    assert isinstance(network.local_balancing_stacks, NetworkElementsDict)
    assert len(network.local_balancing_stacks) == 2
    assert isinstance(network.aggregated_consumers, NetworkElementsDict)
    assert len(network.aggregated_consumers) == 1
    assert isinstance(network.fuels, NetworkElementsDict)
    assert len(network.fuels) == 2
    assert isinstance(network.transmission_fees, NetworkElementsDict)
    assert len(network.transmission_fees) == 2
    assert isinstance(network.emission_fees, NetworkElementsDict)
    assert len(network.emission_fees) == 2


def test_network_creator_create_assertion_occurred() -> None:
    network_constants = NetworkConstants(
        4,
        24,
        {
            CO2_EMISSION: pd.Series([np.nan, 0.95, 0.85, 0.75], index=np.arange(0, 4)),
            PM10_EMISSION: pd.Series([np.nan, 0.95, 0.85, 0.75], index=np.arange(0, 4)),
        },
        {
            CO2_EMISSION: 0.1,
            PM10_EMISSION: 0.15,
        },
        {},
    )
    bus_a = Bus(
        name="bus_A",
        energy_type=ELECTRICITY,
    )
    bus_b = Bus(
        name="bus_B",
        energy_type=HEATING,
    )
    bus_c = Bus(name="bus_C", energy_type=ELECTRICITY)
    bus_d = Bus(name="bus_D", energy_type=HEATING)
    bus_e = Bus(name="bus_E", energy_type=ELECTRICITY)
    gen_type = get_default_generator_type(series_length=network_constants.n_years)
    storage_type = get_default_storage_type(series_length=network_constants.n_years)
    gen_a = Generator(
        name="gen_A",
        energy_source_type=gen_type.name,
        bus={"bus_A", "bus_D"},
        unit_base_cap=30,
        unit_min_capacity=pd.Series([np.nan] * network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * network_constants.n_years),
        unit_min_capacity_increase=pd.Series([np.nan] * network_constants.n_years),
        unit_max_capacity_increase=pd.Series([np.nan] * network_constants.n_years),
    )
    gen_b = Generator(
        name="gen_B",
        energy_source_type=gen_type.name,
        bus={"bus_B", "bus_E"},
        unit_base_cap=10,
        unit_min_capacity=pd.Series([np.nan] * network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * network_constants.n_years),
        unit_min_capacity_increase=pd.Series([np.nan] * network_constants.n_years),
        unit_max_capacity_increase=pd.Series([np.nan] * network_constants.n_years),
    )
    storage = Storage(
        name="storage_A",
        bus="bus_A",
        energy_source_type=storage_type.name,
        unit_base_cap=10,
        unit_min_capacity=pd.Series([np.nan] * network_constants.n_years),
        unit_max_capacity=pd.Series([np.nan] * network_constants.n_years),
        unit_min_capacity_increase=pd.Series([np.nan] * network_constants.n_years),
        unit_max_capacity_increase=pd.Series([np.nan] * network_constants.n_years),
    )
    line_ac = Line(
        name="A->C",
        fr="bus_A",
        to="bus_C",
        transmission_loss=1e-3,
        max_capacity=100.0,
        energy_type=ELECTRICITY,
    )
    lb_stack_cb = LocalBalancingStack(
        name="lb_stack_CB", buses_out={HEATING: "bus_B", ELECTRICITY: "bus_C"}
    )
    lb_stack_ab = LocalBalancingStack(
        name="lb_stack_AB", buses_out={HEATING: "bus_B", ELECTRICITY: "bus_A"}
    )
    aggregate = AggregatedConsumer(
        "aggr_A",
        "default",
        {"lb_stack_AB": 0.3, "lb_stack_CB": 0.7},
        default_yearly_demand,
        {
            "lb_stack_AB": pd.Series([np.nan] * 4, index=range(4)),
            "lb_stack_CB": pd.Series([np.nan] * 4, index=range(4)),
        },
        {
            "lb_stack_AB": pd.Series([np.nan] * 4, index=range(4)),
            "lb_stack_CB": pd.Series([np.nan] * 4, index=range(4)),
        },
        {
            "lb_stack_AB": pd.Series([np.nan] * 4, index=range(4)),
            "lb_stack_CB": pd.Series([np.nan] * 4, index=range(4)),
        },
        {
            "lb_stack_AB": pd.Series([np.nan] * 4, index=range(4)),
            "lb_stack_CB": pd.Series([np.nan] * 4, index=range(4)),
        },
        pd.Series([10000] * 4),
        None,
    )
    coal = Fuel(
        name="coal",
        cost=pd.Series([1] * network_constants.n_years),
        availability=pd.Series([1] * network_constants.n_years),
        emission={},
        energy_per_unit=0.4,
    )
    sun = CapacityFactor(name="sun", profile=pd.Series(data=np.arange(24)))
    demand_profile = DemandProfile("default", default_energy_profile())
    dsr = DSR(
        name="DSR_1",
        compensation_factor=0.1,
        balancing_period_len=20,
        penalization=0.1,
        relative_shift_limit=0.1,
        abs_shift_limit=None,
    )
    capacity_bound = CapacityBound(
        name="Capacity_Bound_gen_A__gen_B",
        left_technology="gen_A",
        right_technology="gen_B",
        sense="EQ",
        left_coefficient=0.4,
    )

    with pytest.raises(NetworkValidatorException) as error_info:
        NetworkCreator._create_network(
            energy_types=[ELECTRICITY, HEATING],
            buses=(bus_a, bus_a, bus_c, bus_d, bus_e),
            generators=(gen_a, gen_b),
            storages=(storage,),
            lines=(line_ac,),
            local_balancing_stacks=(lb_stack_ab, lb_stack_cb),
            aggregated_consumers=(aggregate,),
            fuels=(coal,),
            capacity_factors=(sun,),
            emission_types=[CO2_EMISSION, PM10_EMISSION],
            generator_types=(gen_type,),
            storage_types=(storage_type,),
            demand_profiles=(demand_profile,),
            network_constants=network_constants,
            transmission_fees=(),
            emission_fees=(),
            demand_chunks=(),
            dsr=(dsr,),
            generation_fractions=(),
            capacity_bounds=(capacity_bound,),
        )
    assert (
        str(error_info.value.args[0])
        == "While creating object network following errors occurred: "
    )
    assert (
        str(error_info.value.args[1][0])
        == "Network element Bus with name bus_A has been already added"
    )
    assert (
        str(error_info.value.args[1][1].args[1][0])
        == "Cannot attach generator to a bus bus_B - bus does not exist in the network"
    )
    line_ac = Line(
        name="A->B",
        fr="bus_A",
        to="bus_B",
        transmission_loss=1e-3,
        max_capacity=100.0,
        energy_type=ELECTRICITY,
    )
    em_so2 = EmissionFee(
        name="em_SO2",
        price=pd.Series([0.2] * network_constants.n_years),
        emission_type="SO2",
    )
    dsr = DSR(
        name="DSR_1",
        compensation_factor=0.1,
        balancing_period_len=20,
        penalization=0.1,
        relative_shift_limit=0.1,
        abs_shift_limit=None,
    )
    with pytest.raises(NetworkValidatorException) as error_info:
        NetworkCreator._create_network(
            energy_types=[ELECTRICITY, HEATING],
            buses=(bus_a, bus_b, bus_c, bus_d, bus_e),
            generators=(gen_a, gen_b),
            storages=(storage,),
            lines=(line_ac,),
            local_balancing_stacks=(lb_stack_ab, lb_stack_cb),
            aggregated_consumers=(aggregate,),
            fuels=(coal,),
            capacity_factors=(sun,),
            emission_types=[CO2_EMISSION, PM10_EMISSION],
            generator_types=(gen_type,),
            storage_types=(storage_type,),
            demand_profiles=(demand_profile,),
            network_constants=network_constants,
            transmission_fees=(),
            emission_fees=(em_so2,),
            demand_chunks=(),
            dsr=(dsr,),
            generation_fractions=(),
            capacity_bounds=(),
        )

    assert (
        str(error_info.value.args[0])
        == "While creating object network following errors occurred: "
    )
    assert (
        str(error_info.value.args[1][1].args[1][0])
        == "Cannot set end of the line to bus bus_B. Bus bus_B energy type is HEATING, which is different"
        " from the line energy type: ELECTRICITY."
    )
    assert (
        str(error_info.value.args[1][1].args[1][1])
        == "Cannot add a line between buses bus_A and bus_B with different energy types ELECTRICITY != HEATING"
    )
    assert (
        str(error_info.value.args[1][0].args[1][0])
        == "Emission type: SO2 does not exist in the network"
    )
