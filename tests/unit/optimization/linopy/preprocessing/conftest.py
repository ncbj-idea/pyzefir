from copy import deepcopy
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.network import Network
from pyzefir.model.network_elements import (
    DSR,
    AggregatedConsumer,
    Bus,
    DemandChunk,
    DemandProfile,
    Generator,
    Line,
    LocalBalancingStack,
    Storage,
    TransmissionFee,
)
from tests.unit.optimization.linopy.conftest import (
    N_HOURS,
    N_YEARS,
    generation_fraction,
)
from tests.unit.optimization.linopy.names import EE, GRID, HEAT, HS
from tests.unit.optimization.linopy.utils import add_generators, add_storages, aggr_name

N_LBS = 4


def line_name(bus_fr: str, bus_to: str) -> str:
    return f"{bus_fr}->{bus_to}"


def lbs_name(lbs_idx: int) -> str:
    return f"LBS_{lbs_idx}"


def lbs_bus_name(lbs_idx: int, energy_type: str) -> str:
    return f"{lbs_name(lbs_idx)}_{energy_type}"


@pytest.fixture
def generator_factory() -> Callable[..., Generator]:
    """Wrapper for Generator initialization (with default values provided)"""
    default_capacity_limits = pd.Series(np.ones(N_YEARS) * np.nan)

    def _create_generator(
        name: str,
        bus: str,
        energy_source_type: str | None = None,
        unit_base_cap: float = 10.0,
        emission_fee: set[str] | None = None,
        generator_binding: str | None = None,
        unit_min_capacity: pd.Series = default_capacity_limits,
        unit_max_capacity: pd.Series = default_capacity_limits,
        unit_min_capacity_increase: pd.Series = default_capacity_limits,
        unit_max_capacity_increase: pd.Series = default_capacity_limits,
        min_device_nom_power: float | None = None,
        max_device_nom_power: float | None = None,
        tags: list[str] | None = None,
    ) -> Generator:
        return Generator(
            name=name,
            energy_source_type=energy_source_type or "",
            bus=bus,
            emission_fee=emission_fee or {},
            generator_binding=generator_binding,
            unit_base_cap=unit_base_cap,
            unit_min_capacity=unit_min_capacity,
            unit_max_capacity=unit_max_capacity,
            unit_min_capacity_increase=unit_min_capacity_increase,
            unit_max_capacity_increase=unit_max_capacity_increase,
            min_device_nom_power=min_device_nom_power,
            max_device_nom_power=max_device_nom_power,
            tags=tags or [],
        )

    return _create_generator


@pytest.fixture
def storage_factory() -> Callable[..., Storage]:
    """Wrapper for Storage initialization (with default values provided)"""
    default_capacity_limits = pd.Series(np.ones(N_YEARS) * np.nan)

    def _create_storage(
        name: str,
        bus: str,
        energy_source_type: str | None = None,
        unit_base_cap: float = 10.0,
        unit_min_capacity: pd.Series = default_capacity_limits,
        unit_max_capacity: pd.Series = default_capacity_limits,
        unit_min_capacity_increase: pd.Series = default_capacity_limits,
        unit_max_capacity_increase: pd.Series = default_capacity_limits,
        min_device_nom_power: float | None = None,
        max_device_nom_power: float | None = None,
        tags: list[str] | None = None,
    ) -> Storage:
        return Storage(
            name=name,
            energy_source_type=energy_source_type or "",
            bus=bus,
            unit_base_cap=unit_base_cap,
            unit_min_capacity=unit_min_capacity,
            unit_max_capacity=unit_max_capacity,
            unit_min_capacity_increase=unit_min_capacity_increase,
            unit_max_capacity_increase=unit_max_capacity_increase,
            min_device_nom_power=min_device_nom_power,
            max_device_nom_power=max_device_nom_power,
            tags=tags or [],
        )

    return _create_storage


@pytest.fixture
def demand_chunk_factory() -> Callable[..., DemandChunk]:
    """Wrapper for DemandChunk initialization (with default values provided)"""

    def _create_demand_chunk(
        name: str,
        tag: str,
        energy_type: str,
        n_years: int | None = None,
        n_hours: int | None = None,
        periods: np.ndarray | None = None,
        demand: np.ndarray | None = None,
    ) -> DemandChunk:
        n_years = n_years or N_YEARS
        n_hours = n_hours or N_HOURS
        default_periods = np.array([[0, n_hours - 1]])
        default_demand = np.array([[10 for _ in range(n_years)]])
        return DemandChunk(
            name=name,
            tag=tag,
            energy_type=energy_type,
            periods=periods or default_periods,
            demand=demand or default_demand,
        )

    return _create_demand_chunk


@pytest.fixture
def dsr_factory() -> Callable[..., DSR]:
    """Wrapper for DSR initialization (with default values provided)"""

    def _create_dsr(
        name: str,
        compensation_factor: float = 1.0,
        balancing_period_len: int = 24,
        penalization: float = 1.0,
        penalization_plus: float = 0.0,
        relative_shift_limit: float | None = None,
        abs_shift_limit: float | None = None,
    ) -> DSR:
        return DSR(
            name=name,
            compensation_factor=compensation_factor,
            balancing_period_len=balancing_period_len,
            penalization_minus=penalization,
            penalization_plus=penalization_plus,
            relative_shift_limit=relative_shift_limit,
            abs_shift_limit=abs_shift_limit,
        )

    return _create_dsr


@pytest.fixture
def global_generators(
    generator_factory: Callable[..., Generator]
) -> dict[str, Generator]:
    return {
        f"heat_pump_{GRID}_{HS}": generator_factory(
            name=f"heat_pump_{GRID}_{HS}",
            energy_source_type="heat_pump",
            unit_base_cap=30,
            bus={HS, GRID},
        ),
        f"chp_coal_{GRID}_{HS}": generator_factory(
            name=f"chp_coal_{GRID}_{HS}",
            energy_source_type="chp_coal",
            unit_base_cap=50,
            bus={GRID, HS},
            unit_max_capacity_increase=pd.Series(
                [np.nan] + np.linspace(1, 1.8, N_YEARS - 1).tolist()
            ),
        ),
        f"pp_coal_{GRID}": generator_factory(
            name=f"pp_coal_{GRID}",
            energy_source_type="pp_coal",
            unit_base_cap=50,
            bus=GRID,
            min_device_nom_power=1.0,
            max_device_nom_power=5.0,
            unit_max_capacity_increase=pd.Series(
                [np.nan] + np.linspace(1, 1.8, N_YEARS - 1).tolist()
            ),
        ),
        f"pp_gas_{GRID}": generator_factory(
            name=f"pp_gas_{GRID}",
            energy_source_type="pp_gas",
            unit_base_cap=40,
            bus=GRID,
            min_device_nom_power=2.3,
            max_device_nom_power=5.6,
        ),
        f"heat_plant_coal_{HS}": generator_factory(
            name=f"heat_plant_coal_{HS}",
            energy_source_type="heat_plant_coal",
            unit_base_cap=30,
            bus=HS,
            unit_min_capacity_increase=pd.Series(
                [np.nan] + np.linspace(1, 3, N_YEARS - 1).tolist()
            ),
        ),
        f"heat_plant_biomass_{HS}": generator_factory(
            name=f"heat_plant_biomass_{HS}",
            energy_source_type="heat_plant_biomass",
            unit_base_cap=45,
            bus=HS,
        ),
    }


@pytest.fixture
def global_storages(storage_factory: Callable[..., Storage]) -> dict[str, Storage]:
    return {
        f"heat_storage_{HS}": storage_factory(
            name=f"heat_storage_{HS}",
            energy_source_type="heat_storage_type",
            unit_base_cap=15,
            bus=HS,
        ),
        f"heat_storage_{HS}_2": storage_factory(
            name=f"heat_storage_{HS}_2",
            energy_source_type="heat_storage_type",
            unit_base_cap=15,
            bus=HS,
            min_device_nom_power=1.0,
            max_device_nom_power=3.0,
        ),
    }


@pytest.fixture
def lbs_generators(
    generator_factory: Callable[..., Generator]
) -> dict[int, dict[str, Generator]]:
    return {
        0: {
            f"boiler_coal_{lbs_name(0)}": generator_factory(
                name=f"boiler_coal_{lbs_name(0)}",
                energy_source_type="boiler_coal",
                unit_base_cap=25,
                bus=lbs_bus_name(0, HEAT),
                unit_max_capacity=pd.Series([np.nan] + [25] * (N_YEARS - 1)),
                tags=["example_tag"],
            ),
        },
        1: {
            f"solar_{lbs_name(1)}": generator_factory(
                name=f"solar_{lbs_name(1)}",
                energy_source_type="solar",
                unit_base_cap=10,
                bus=lbs_bus_name(1, HEAT),
                unit_max_capacity=pd.Series([np.nan] + [25] * (N_YEARS - 1)),
                tags=["example_sub_tag"],
            ),
            f"wind_farm_{lbs_name(1)}": generator_factory(
                name=f"wind_farm_{lbs_name(1)}",
                energy_source_type="wind_farm",
                unit_base_cap=10,
                bus=lbs_bus_name(1, EE),
                unit_min_capacity=pd.Series(
                    [np.nan] + np.linspace(10, 20, N_YEARS - 1).tolist()
                ),
                unit_max_capacity=pd.Series([np.nan] + [25] * (N_YEARS - 1)),
            ),
        },
        2: {
            f"heat_pump_{lbs_name(2)}": generator_factory(
                name=f"heat_pump_{lbs_name(2)}",
                energy_source_type="heat_pump",
                unit_base_cap=30,
                bus={lbs_bus_name(2, HEAT), lbs_bus_name(2, EE)},
                unit_max_capacity_increase=pd.Series([np.nan] + [2] * (N_YEARS - 1)),
                unit_max_capacity=pd.Series([np.nan] + [25] * (N_YEARS - 1)),
            ),
            f"pv_{lbs_name(2)}": generator_factory(
                name=f"pv_{lbs_name(2)}",
                energy_source_type="pv",
                unit_base_cap=15,
                bus=lbs_bus_name(2, EE),
                unit_max_capacity=pd.Series([np.nan] + [25] * (N_YEARS - 1)),
            ),
        },
        3: {},
    }


@pytest.fixture
def lbs_storages(
    storage_factory: Callable[..., Storage]
) -> dict[int, dict[str, Storage]]:
    return {
        0: {},
        1: {
            f"heat_storage_{lbs_name(1)}": storage_factory(
                name=f"heat_storage_{lbs_name(1)}",
                energy_source_type="heat_storage_type",
                unit_base_cap=15,
                bus=lbs_bus_name(1, HEAT),
                unit_max_capacity=pd.Series([np.nan] + [25] * (N_YEARS - 1)),
            ),
            f"ee_storage_{lbs_name(1)}": storage_factory(
                name=f"ee_storage_{lbs_name(1)}",
                energy_source_type="ee_storage_type",
                unit_base_cap=12,
                bus=lbs_bus_name(1, EE),
                unit_max_capacity=pd.Series([np.nan] + [25] * (N_YEARS - 1)),
            ),
        },
        2: {
            f"heat_storage_{lbs_name(2)}": storage_factory(
                name=f"heat_storage_{lbs_name(2)}",
                energy_source_type="heat_storage_type",
                unit_base_cap=10,
                bus=lbs_bus_name(2, HEAT),
                unit_max_capacity=pd.Series([np.nan] + [25] * (N_YEARS - 1)),
            ),
        },
        3: {},
    }


@pytest.fixture
def lbs_bus_factory() -> Callable[..., Bus]:
    def _create_lbs_bus(lbs_idx: int, energy_type: str) -> Bus:
        return Bus(
            name=lbs_bus_name(lbs_idx, energy_type),
            energy_type=energy_type,
        )

    return _create_lbs_bus


@pytest.fixture
def lbs_factory() -> Callable[[int], LocalBalancingStack]:
    def _create_lbs(lbs_idx: int) -> LocalBalancingStack:
        return LocalBalancingStack(
            name=lbs_name(lbs_idx),
            buses_out={
                HEAT: lbs_bus_name(lbs_idx, HEAT),
                EE: lbs_bus_name(lbs_idx, EE),
            },
            buses={
                EE: {lbs_bus_name(lbs_idx, EE)},
                HEAT: {lbs_bus_name(lbs_idx, HEAT)},
            },
        )

    return _create_lbs


@pytest.fixture
def create_lbs_kse_line() -> Callable[..., Line]:
    def _lbs_kse_line(
        lbs_idx: int,
        transmission_loss: float = 0.03,
        max_capacity: float = np.infty,
        transmission_fee: str | None = None,
    ) -> Line:
        return Line(
            name=line_name(lbs_bus_name(lbs_idx, EE), GRID),
            energy_type=EE,
            fr=lbs_bus_name(lbs_idx, EE),
            to=GRID,
            max_capacity=max_capacity,
            transmission_loss=transmission_loss,
            transmission_fee=transmission_fee,
        )

    return _lbs_kse_line


@pytest.fixture
def create_lbs_hs_line() -> Callable[..., Line]:
    def _lbs_hs_line(
        lbs_idx: int, transmission_loss: float = 0.03, max_capacity: float = np.infty
    ) -> Line:
        return Line(
            name=line_name(lbs_bus_name(lbs_idx, HEAT), HS),
            energy_type=HEAT,
            fr=lbs_bus_name(lbs_idx, HEAT),
            to=HS,
            max_capacity=max_capacity,
            transmission_loss=transmission_loss,
        )

    return _lbs_hs_line


@pytest.fixture
def aggr_factory(demand_profile: DemandProfile) -> Callable[..., AggregatedConsumer]:
    default_energy_usage = {
        HEAT: pd.Series(data=np.linspace(1e6, 0.8 * 1e6, N_YEARS)),
        EE: pd.Series(data=np.linspace(1e6, 1.1 * 1e6, N_YEARS)),
    }

    def _create_aggr(
        aggr_idx: int,
        base_fractions: dict[int, float],
        fractions: dict[int, pd.Series],
        yearly_energy_usage: dict[str, pd.Series] | None = None,
    ) -> AggregatedConsumer:
        return AggregatedConsumer(
            name=aggr_name(aggr_idx),
            demand_profile=demand_profile.name,
            stack_base_fraction={
                lbs_name(lbs_idx): base_fraction
                for lbs_idx, base_fraction in base_fractions.items()
            },
            yearly_energy_usage=yearly_energy_usage or default_energy_usage,
            min_fraction={
                lbs_name(lbs_idx): fraction_series
                for lbs_idx, fraction_series in fractions.items()
            },
            max_fraction={
                lbs_name(lbs_idx): fraction_series
                for lbs_idx, fraction_series in fractions.items()
            },
            max_fraction_decrease={
                lbs_name(lbs_idx): fraction_series
                for lbs_idx, fraction_series in fractions.items()
            },
            max_fraction_increase={
                lbs_name(lbs_idx): fraction_series
                for lbs_idx, fraction_series in fractions.items()
            },
            n_consumers=pd.Series([1000] * N_YEARS),
            average_area=None,
        )

    return _create_aggr


@pytest.fixture
def complete_network(
    empty_network: Network,
    grid_bus: Bus,
    hs_bus: Bus,
    lbs_bus_factory: Callable[..., Bus],
    global_generators: dict[str, Generator],
    global_storages: dict[str, Storage],
    lbs_generators: dict[int, dict[str, Generator]],
    lbs_storages: dict[int, dict[str, Storage]],
    demand_profile: DemandProfile,
    create_lbs_kse_line: Callable[..., Line],
    create_lbs_hs_line: Callable[..., Line],
    lbs_factory: Callable[..., LocalBalancingStack],
    aggr_factory: Callable[..., AggregatedConsumer],
    generation_fraction: generation_fraction,
) -> Network:
    _network = deepcopy(empty_network)
    _network.add_bus(grid_bus)
    _network.add_bus(hs_bus)

    add_generators(_network, global_generators.values())
    add_storages(_network, global_storages.values())

    for lbs_idx in range(N_LBS):
        _network.add_bus(lbs_bus_factory(lbs_idx, HEAT))
        _network.add_bus(lbs_bus_factory(lbs_idx, EE))
        add_generators(_network, lbs_generators[lbs_idx].values())
        add_storages(_network, lbs_storages[lbs_idx].values())
        _network.add_transmission_fee(
            TransmissionFee(
                name=f"tf_{lbs_idx}", fee=pd.Series(data=[lbs_idx / 10] * N_HOURS)
            )
        )
        _network.add_line(
            create_lbs_kse_line(lbs_idx, transmission_fee=f"tf_{lbs_idx}")
        )
        _network.add_local_balancing_stack(lbs_factory(lbs_idx))

    for hs_lbs_idx in [1, 3]:
        _network.add_line(create_lbs_hs_line(hs_lbs_idx))

    _network.add_demand_profile(demand_profile)
    _network.add_aggregated_consumer(
        aggr_factory(
            aggr_idx=0,
            base_fractions={0: 0.3, 1: 0.7},
            fractions={
                0: pd.Series([np.nan] * N_YEARS),
                1: pd.Series([np.nan] * N_YEARS),
            },
        )
    )
    _network.add_aggregated_consumer(
        aggr_factory(
            aggr_idx=1,
            base_fractions={2: 0.4, 3: 0.6},
            fractions={
                2: pd.Series([np.nan] * N_YEARS),
                3: pd.Series([np.nan] * N_YEARS),
            },
        )
    )
    _network.add_generation_fraction(generation_fraction)

    return _network
