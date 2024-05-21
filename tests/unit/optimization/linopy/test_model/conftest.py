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

from pyzefir.model.network_elements import (
    AggregatedConsumer,
    Bus,
    DemandProfile,
    Generator,
    Line,
    LocalBalancingStack,
    Storage,
    TransmissionFee,
)
from pyzefir.model.network_elements.emission_fee import EmissionFee
from tests.unit.optimization.linopy.constants import N_HOURS, N_YEARS
from tests.unit.optimization.linopy.names import CO2, EE, GRID, HEAT, HS, PM10


@pytest.fixture
def local_ee_bus() -> Bus:
    """
    Local electric bus (in local balancing stack lbs)
    """
    return Bus(name="local_ee_bus", energy_type=EE)


@pytest.fixture
def local_ee_bus2() -> Bus:
    """
    Local electric bus (in local balancing stack lbs)
    """
    return Bus(name="local_ee_bus2", energy_type=EE)


@pytest.fixture
def local_heat_bus() -> Bus:
    """
    Local heat bus (in local balancing stack lbs)
    """
    return Bus(name="local_heat_bus", energy_type=HEAT)


@pytest.fixture
def local_heat_bus2() -> Bus:
    """
    Local heat bus (in local balancing stack lbs)
    """
    return Bus(name="local_heat_bus2", energy_type=HEAT)


@pytest.fixture
def transmission_fee() -> TransmissionFee:
    """Transmission fee for each hour of the year."""
    return TransmissionFee(name="TransmissionFee", fee=pd.Series([0.19] * N_HOURS))


@pytest.fixture
def emission_fee_CO2() -> EmissionFee:
    """Emission Fee for CO2"""
    return EmissionFee(
        name="CO2_EMF", emission_type=CO2, price=pd.Series([0.0] * N_YEARS)
    )


@pytest.fixture
def emission_fee_PM10() -> EmissionFee:
    """Emission Fee for PM10"""
    return EmissionFee(
        name="PM10_EMF", emission_type=PM10, price=pd.Series([0.0] * N_YEARS)
    )


@pytest.fixture
def grid_connection(
    local_ee_bus: Bus, grid_bus: Bus, transmission_fee: TransmissionFee
) -> Line:
    """
    Line connecting the local_ee_bus with grid (grid_bus)
    """
    return Line(
        name=f"{grid_bus.name}->{local_ee_bus.name}",
        energy_type=EE,
        fr=grid_bus.name,
        to=local_ee_bus.name,
        transmission_loss=0,
        max_capacity=np.infty,
        transmission_fee=transmission_fee.name,
    )


@pytest.fixture
def grid_connection_no_fee(local_ee_bus: Bus, grid_bus: Bus) -> Line:
    """
    Line connecting the local_ee_bus with grid (grid_bus)
    """
    return Line(
        name=f"{grid_bus.name}->{local_ee_bus.name}",
        energy_type=EE,
        fr=grid_bus.name,
        to=local_ee_bus.name,
        transmission_loss=0,
        max_capacity=np.infty,
    )


@pytest.fixture
def grid_connection2(
    local_ee_bus2: Bus, grid_bus: Bus, transmission_fee: TransmissionFee
) -> Line:
    """
    Line connecting the local_ee_bus with grid (grid_bus)
    """
    return Line(
        name=f"{grid_bus.name}->{local_ee_bus2.name}",
        energy_type=EE,
        fr=grid_bus.name,
        to=local_ee_bus2.name,
        transmission_loss=0,
        max_capacity=np.infty,
        transmission_fee=transmission_fee.name,
    )


@pytest.fixture
def heating_system_connection(local_heat_bus: Bus, hs_bus: Bus) -> Line:
    """
    Line connecting the local_heat_bus with heating system (hs_bus)
    """
    return Line(
        name=f"{hs_bus.name}->{local_heat_bus.name}",
        energy_type=HEAT,
        fr=hs_bus.name,
        to=local_heat_bus.name,
        transmission_loss=0,
        max_capacity=np.infty,
    )


@pytest.fixture
def heating_system_connection2(local_heat_bus2: Bus, hs_bus: Bus) -> Line:
    """
    Line connecting the local_heat_bus with heating system (hs_bus)
    """
    return Line(
        name=f"{hs_bus.name}->{local_heat_bus2.name}",
        energy_type=HEAT,
        fr=hs_bus.name,
        to=local_heat_bus2.name,
        transmission_loss=0,
        max_capacity=np.infty,
    )


@pytest.fixture
def coal_power_plant(grid_bus: Bus) -> Generator:
    """
    Power plant generator connected to the grid (grid_bus)
    """
    return Generator(
        name=f"pp_coal_{GRID}",
        energy_source_type="pp_coal",
        bus=grid_bus.name,
        unit_base_cap=40,
        unit_min_capacity=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity=pd.Series([np.nan] * N_YEARS),
        unit_min_capacity_increase=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity_increase=pd.Series([np.nan] * N_YEARS),
    )


@pytest.fixture
def coal_chp(grid_bus: Bus, hs_bus: Bus) -> Generator:
    """
    CHP generator connected to the grid (grid_bus) and heating system (hs_bus).
    """
    return Generator(
        name=f"chp_coal_{grid_bus.name}_{hs_bus.name}",
        energy_source_type="chp_coal",
        bus={grid_bus.name, hs_bus.name},
        unit_base_cap=1500,
        unit_min_capacity=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity=pd.Series([np.nan] * N_YEARS),
        unit_min_capacity_increase=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity_increase=pd.Series([np.nan] * N_YEARS),
    )


@pytest.fixture
def biomass_heat_plant(hs_bus: Bus) -> Generator:
    """
    Biomass heat plant generator connected to the heating system (hs_bus)
    """
    return Generator(
        name=f"biomass_heat_plant_{HS}",
        energy_source_type="heat_plant_biomass",
        bus=hs_bus.name,
        unit_base_cap=30,
        unit_min_capacity=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity=pd.Series([np.nan] * N_YEARS),
        unit_min_capacity_increase=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity_increase=pd.Series([np.nan] * N_YEARS),
    )


@pytest.fixture
def coal_heat_plant(hs_bus: Bus) -> Generator:
    """
    Coal heat plant generator connected to the heating system (hs_bus)
    """
    return Generator(
        name=f"coal_heat_plant_{HS}",
        energy_source_type="heat_plant_coal",
        bus=hs_bus.name,
        unit_base_cap=30,
        unit_min_capacity=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity=pd.Series([np.nan] * N_YEARS),
        unit_min_capacity_increase=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity_increase=pd.Series([np.nan] * N_YEARS),
    )


@pytest.fixture
def local_coal_heat_plant(local_heat_bus: Bus) -> Generator:
    """
    local heat plant coal
    """
    return Generator(
        name="local_coal_heat_plant",
        energy_source_type="local_coal_heat_plant",
        bus=local_heat_bus.name,
        unit_base_cap=17500,
        unit_min_capacity=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity=pd.Series([np.nan] * N_YEARS),
        unit_min_capacity_increase=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity_increase=pd.Series([np.nan] * N_YEARS),
    )


@pytest.fixture
def local_coal_heat_plant2(local_heat_bus2: Bus) -> Generator:
    """
    local heat plant coal (v2)
    """
    return Generator(
        name="local_coal_heat_plant2",
        energy_source_type="local_coal_heat_plant2",
        bus=local_heat_bus2.name,
        unit_base_cap=17500,
        unit_min_capacity=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity=pd.Series([np.nan] * N_YEARS),
        unit_min_capacity_increase=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity_increase=pd.Series([np.nan] * N_YEARS),
    )


@pytest.fixture
def local_pv(local_ee_bus: Bus) -> Generator:
    """Local pv generator."""
    return Generator(
        name="local_pv",
        energy_source_type="pv",
        bus={local_ee_bus.name},
        unit_base_cap=30,
        unit_min_capacity=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity=pd.Series([np.nan] * N_YEARS),
        unit_min_capacity_increase=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity_increase=pd.Series([np.nan] * N_YEARS),
    )


@pytest.fixture
def local_pv2(local_ee_bus2: Bus) -> Generator:
    """Local pv generator."""
    return Generator(
        name="local_pv2",
        energy_source_type="pv",
        bus={"local_ee_bus2"},
        unit_base_cap=35,
        unit_min_capacity=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity=pd.Series([np.nan] * N_YEARS),
        unit_min_capacity_increase=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity_increase=pd.Series([np.nan] * N_YEARS),
    )


@pytest.fixture
def global_solar(hs_bus: Bus) -> Generator:
    return Generator(
        name="global_solar",
        energy_source_type="solar",
        bus={hs_bus.name},
        unit_base_cap=10,
        unit_min_capacity=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity=pd.Series([np.nan] * N_YEARS),
        unit_min_capacity_increase=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity_increase=pd.Series([np.nan] * N_YEARS),
    )


@pytest.fixture
def heat_storage(hs_bus: Bus) -> Storage:
    return Storage(
        name="heat_storage",
        energy_source_type="heat_storage_type",
        unit_base_cap=15,
        bus=hs_bus.name,
        unit_min_capacity=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity=pd.Series([np.nan] * N_YEARS),
        unit_min_capacity_increase=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity_increase=pd.Series([np.nan] * N_YEARS),
    )


@pytest.fixture
def ee_storage(grid_bus: Bus) -> Storage:
    return Storage(
        name="ee_storage",
        energy_source_type="ee_storage_type",
        unit_base_cap=15,
        bus=grid_bus.name,
        unit_min_capacity=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity=pd.Series([np.nan] * N_YEARS),
        unit_min_capacity_increase=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity_increase=pd.Series([np.nan] * N_YEARS),
    )


@pytest.fixture
def lbs(local_heat_bus: Bus, local_ee_bus: Bus) -> LocalBalancingStack:
    """
    Local balancing stack connected to the grid and heating system (no local energy sources)
    """
    return LocalBalancingStack(
        name="lbs",
        buses_out={HEAT: local_heat_bus.name, EE: local_ee_bus.name},
        buses={EE: {"local_ee_bus"}, HEAT: {"local_heat_bus"}},
    )


@pytest.fixture
def lbs_ee(local_ee_bus: Bus) -> LocalBalancingStack:
    """
    Local balancing stack connected to the grid and heating system (no local energy sources)
    """
    return LocalBalancingStack(
        name="lbs_ee",
        buses_out={EE: "local_ee_bus"},
        buses={EE: {"local_ee_bus"}},
    )


@pytest.fixture
def lbs2(local_heat_bus: Bus, local_ee_bus: Bus) -> LocalBalancingStack:
    """
    Local balancing stack connected to the grid and heating system (no local energy sources)
    """
    return LocalBalancingStack(
        name="lbs2",
        buses_out={HEAT: local_heat_bus.name, EE: local_ee_bus.name},
        buses={EE: {"local_ee_bus2"}, HEAT: {"local_heat_bus2"}},
    )


@pytest.fixture
def lbs_ee2(local_heat_bus: Bus, local_ee_bus: Bus) -> LocalBalancingStack:
    """
    Local balancing stack connected to the grid and heating system (no local energy sources)
    """
    return LocalBalancingStack(
        name="lbs_ee2",
        buses_out={EE: "local_ee_bus2"},
        buses={EE: {"local_ee_bus2"}},
    )


@pytest.fixture
def aggr(demand_profile: DemandProfile, lbs: LocalBalancingStack) -> AggregatedConsumer:
    """
    Aggregated consumer, which is using only local balancing stack lbs
    """
    return AggregatedConsumer(
        name="aggr",
        demand_profile=demand_profile.name,
        stack_base_fraction={lbs.name: 1.0},
        yearly_energy_usage={
            HEAT: pd.Series(np.linspace(1e3, 0.85 * 1e3, N_YEARS)),
            EE: pd.Series(np.linspace(1e3, 1.2 * 1e3, N_YEARS)),
        },
        max_fraction={lbs.name: pd.Series([np.nan] * N_YEARS)},
        max_fraction_decrease={lbs.name: pd.Series([np.nan] * N_YEARS)},
        max_fraction_increase={lbs.name: pd.Series([np.nan] * N_YEARS)},
        min_fraction={lbs.name: pd.Series([np.nan] * N_YEARS)},
        n_consumers=pd.Series([1] * N_YEARS),
        average_area=None,
    )


@pytest.fixture
def aggr_ee(
    demand_profile: DemandProfile, lbs: LocalBalancingStack
) -> AggregatedConsumer:
    """
    Aggregated consumer, which is using only local balancing stack lbs
    """
    return AggregatedConsumer(
        name="aggr_ee",
        demand_profile=demand_profile.name,
        stack_base_fraction={"lbs_ee": 1.0, "lbs_ee2": 0.0},
        yearly_energy_usage={
            EE: pd.Series(np.linspace(1e3, 1.2 * 1e3, N_YEARS)),
        },
        max_fraction={
            "lbs_ee": pd.Series([np.nan] * N_YEARS),
            "lbs_ee2": pd.Series([np.nan] * N_YEARS),
        },
        max_fraction_decrease={
            "lbs_ee": pd.Series([np.nan] * N_YEARS),
            "lbs_ee2": pd.Series([np.nan] * N_YEARS),
        },
        max_fraction_increase={
            "lbs_ee": pd.Series([np.nan] * N_YEARS),
            "lbs_ee2": pd.Series([np.nan] * N_YEARS),
        },
        min_fraction={
            "lbs_ee": pd.Series([np.nan] * N_YEARS),
            "lbs_ee2": pd.Series([np.nan] * N_YEARS),
        },
        n_consumers=pd.Series([1] * N_YEARS),
        average_area=None,
    )


@pytest.fixture
def aggr_b(
    demand_profile: DemandProfile,
    lbs_b: LocalBalancingStack,
    lbs2_b: LocalBalancingStack,
) -> AggregatedConsumer:
    """
    Aggregated consumer, which is using only local balancing stack lbs
    """
    return AggregatedConsumer(
        name="aggr_b",
        demand_profile=demand_profile.name,
        stack_base_fraction={"lbs_b": 0.3, "lbs2_b": 0.7},
        yearly_energy_usage={
            HEAT: pd.Series(np.linspace(1e3, 0.9 * 1e3, N_YEARS)),
            EE: pd.Series(np.linspace(1e3, 1.2 * 1e3, N_YEARS)),
        },
        max_fraction={
            lbs_b.name: pd.Series([np.nan] * N_YEARS),
            lbs2_b.name: pd.Series([np.nan] * N_YEARS),
        },
        max_fraction_decrease={
            lbs_b.name: pd.Series([np.nan] * N_YEARS),
            lbs2_b.name: pd.Series([np.nan] * N_YEARS),
        },
        max_fraction_increase={
            lbs_b.name: pd.Series([np.nan] * N_YEARS),
            lbs2_b.name: pd.Series([np.nan] * N_YEARS),
        },
        min_fraction={
            lbs_b.name: pd.Series([np.nan] * N_YEARS),
            lbs2_b.name: pd.Series([np.nan] * N_YEARS),
        },
        n_consumers=pd.Series([1000] * N_YEARS),
        average_area=None,
    )


@pytest.fixture
def local_ee_bus_b() -> Bus:
    """
    Local electric bus (in local balancing stack lbs)
    """
    return Bus(name="local_ee_bus_b", energy_type=EE)


@pytest.fixture
def local_ee_bus2_b() -> Bus:
    """
    Local electric bus (in local balancing stack lbs)
    """
    return Bus(name="local_ee_bus2_b", energy_type=EE)


@pytest.fixture
def local_heat_bus_b() -> Bus:
    """
    Local heat bus (in local balancing stack lbs)
    """
    return Bus(name="local_heat_bus_b", energy_type=HEAT)


@pytest.fixture
def local_heat_bus2_b() -> Bus:
    """
    Local heat bus (in local balancing stack lbs)
    """
    return Bus(name="local_heat_bus2_b", energy_type=HEAT)


@pytest.fixture
def grid_connection_b(
    local_ee_bus_b: Bus, grid_bus: Bus, transmission_fee: TransmissionFee
) -> Line:
    """
    Line connecting the local_ee_bus with grid (grid_bus)
    """
    return Line(
        name=f"{grid_bus.name}->{local_ee_bus_b.name}",
        energy_type=EE,
        fr=grid_bus.name,
        to=local_ee_bus_b.name,
        transmission_loss=0,
        max_capacity=np.infty,
        transmission_fee=transmission_fee.name,
    )


@pytest.fixture
def grid_connection2_b(
    local_ee_bus2_b: Bus, grid_bus: Bus, transmission_fee: TransmissionFee
) -> Line:
    """
    Line connecting the local_ee_bus with grid (grid_bus)
    """
    return Line(
        name=f"{grid_bus.name}->{local_ee_bus2_b.name}",
        energy_type=EE,
        fr=grid_bus.name,
        to=local_ee_bus2_b.name,
        transmission_loss=0,
        max_capacity=np.infty,
        transmission_fee=transmission_fee.name,
    )


@pytest.fixture
def heating_system_connection_b(local_heat_bus_b: Bus, hs_bus: Bus) -> Line:
    """
    Line connecting the local_heat_bus with heating system (hs_bus)
    """
    return Line(
        name=f"{hs_bus.name}->{local_heat_bus_b.name}",
        energy_type=HEAT,
        fr=hs_bus.name,
        to=local_heat_bus_b.name,
        transmission_loss=0,
        max_capacity=np.infty,
    )


@pytest.fixture
def heating_system_connection2_b(local_heat_bus2_b: Bus, hs_bus: Bus) -> Line:
    """
    Line connecting the local_heat_bus with heating system (hs_bus)
    """
    return Line(
        name=f"{hs_bus.name}->{local_heat_bus2_b.name}",
        energy_type=HEAT,
        fr=hs_bus.name,
        to=local_heat_bus2_b.name,
        transmission_loss=0,
        max_capacity=np.infty,
    )


@pytest.fixture
def local_coal_heat_plant_b(local_heat_bus_b: Bus) -> Generator:
    """
    local heat plant coal
    """
    return Generator(
        name="local_coal_heat_plant_b",
        energy_source_type="local_coal_heat_plant",
        bus=local_heat_bus_b.name,
        unit_base_cap=17500,
        unit_min_capacity=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity=pd.Series([np.nan] * N_YEARS),
        unit_min_capacity_increase=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity_increase=pd.Series([np.nan] * N_YEARS),
    )


@pytest.fixture
def local_coal_heat_plant2_b(local_heat_bus2_b: Bus) -> Generator:
    """
    local heat plant coal (v2)
    """
    return Generator(
        name="local_coal_heat_plant2_b",
        energy_source_type="local_coal_heat_plant2",
        bus=local_heat_bus2_b.name,
        unit_base_cap=22500,
        unit_min_capacity=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity=pd.Series([np.nan] * N_YEARS),
        unit_min_capacity_increase=pd.Series([np.nan] * N_YEARS),
        unit_max_capacity_increase=pd.Series([np.nan] * N_YEARS),
    )


@pytest.fixture
def lbs_b(local_heat_bus_b: Bus, local_ee_bus: Bus) -> LocalBalancingStack:
    """
    Local balancing stack connected to the grid and heating system (no local energy sources)
    """
    return LocalBalancingStack(
        name="lbs_b",
        buses_out={HEAT: "local_heat_bus_b", EE: "local_ee_bus_b"},
        buses={EE: {"local_ee_bus_b"}, HEAT: {"local_heat_bus_b"}},
    )


@pytest.fixture
def lbs2_b(local_heat_bus_b: Bus, local_ee_bus: Bus) -> LocalBalancingStack:
    """
    Local balancing stack connected to the grid and heating system (no local energy sources)
    """
    return LocalBalancingStack(
        name="lbs2_b",
        buses_out={HEAT: "local_heat_bus2_b", EE: "local_ee_bus2_b"},
        buses={EE: {"local_ee_bus2_b"}, HEAT: {"local_heat_bus2_b"}},
    )
