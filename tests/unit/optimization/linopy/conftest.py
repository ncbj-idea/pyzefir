from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas import Series

from pyzefir.model.network import Network
from pyzefir.model.network_elements import (
    Bus,
    CapacityFactor,
    DemandProfile,
    Fuel,
    GenerationFraction,
    GeneratorType,
    StorageType,
)
from pyzefir.model.utils import NetworkConstants
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.constants import (
    BASE_TOTAL_EMISSION,
    N_HOURS,
    N_YEARS,
    REL_EM_LIM,
)
from tests.unit.optimization.linopy.names import CO2, EE, GRID, HEAT, HS, PM10
from tests.unit.optimization.linopy.preprocessing.utils import (
    create_generator_type,
    create_storage_type,
)


@pytest.fixture
def grid_bus() -> Bus:
    return Bus(
        name=GRID,
        energy_type=EE,
    )


@pytest.fixture
def hs_bus() -> Bus:
    return Bus(
        name=HS,
        energy_type=HEAT,
    )


@pytest.fixture
def insolation_factor(test_network_assets_path: Path) -> pd.Series:
    return pd.read_csv(
        test_network_assets_path / "sun_capacity_factor.csv",
        sep=";",
        header=None,
        index_col=None,
    ).squeeze()


@pytest.fixture
def wind_capacity_factor(test_network_assets_path: Path) -> pd.Series:
    return pd.read_csv(
        test_network_assets_path / "wind_capacity_factor.csv",
        sep=";",
        header=None,
        index_col=None,
    ).squeeze()


@pytest.fixture
def cop(test_network_assets_path: Path) -> pd.Series:
    return (
        1
        / pd.read_csv(
            test_network_assets_path / "cop.csv", sep=";", header=None, index_col=None
        ).squeeze()
    )


@pytest.fixture
def demand_profile(test_network_assets_path: Path) -> DemandProfile:
    return DemandProfile(
        name="multi_family_profile",
        normalized_profile={
            HEAT: pd.read_csv(
                test_network_assets_path / "heat_profile.csv",
                sep=",",
                header=None,
                index_col=0,
            ).squeeze(),
            EE: pd.read_csv(
                test_network_assets_path / "ee_profile.csv",
                sep=",",
                header=None,
                index_col=0,
            ).squeeze(),
        },
    )


@pytest.fixture
def demand_profile_ee(test_network_assets_path: Path) -> DemandProfile:
    return DemandProfile(
        name="multi_family_profile",
        normalized_profile={
            EE: pd.read_csv(
                test_network_assets_path / "ee_profile.csv",
                sep=",",
                header=None,
                index_col=0,
            ).squeeze(),
        },
    )


@pytest.fixture
def network_constants() -> NetworkConstants:
    return NetworkConstants(
        n_years=N_YEARS,
        n_hours=N_HOURS,
        relative_emission_limits=REL_EM_LIM,
        base_total_emission=BASE_TOTAL_EMISSION,
        power_reserves={},
    )


@pytest.fixture
def opt_config() -> OptConfig:
    return OptConfig(
        hours=N_HOURS, years=N_YEARS, discount_rate=np.linspace(0.5, 0.8, N_YEARS)
    )


@pytest.fixture
def fuels() -> dict[str, Fuel]:
    return {
        "coal": Fuel(
            name="coal",
            emission={CO2: 13, PM10: 9},
            cost=pd.Series(data=np.linspace(200, 270, N_YEARS)),
            availability=pd.Series(data=np.linspace(1e6, 1e6 * 0.7, N_YEARS)),
            energy_per_unit=15.0,
        ),
        "biomass": Fuel(
            name="biomass",
            emission={CO2: 0.0, PM10: 0.2},
            availability=pd.Series(data=np.ones(N_YEARS) * np.inf),
            cost=pd.Series(data=np.linspace(75, 55, N_YEARS)),
            energy_per_unit=0.7,
        ),
        "gas": Fuel(
            name="gas",
            emission={CO2: 7, PM10: 0.0},
            cost=pd.Series(data=np.linspace(150, 220, N_YEARS)),
            availability=pd.Series(data=np.linspace(1e6, 1e6 * 1.2, N_YEARS)),
            energy_per_unit=11.0,
        ),
    }


@pytest.fixture
def cfs(
    insolation_factor: pd.Series, wind_capacity_factor: pd.Series
) -> dict[str, CapacityFactor]:
    return {
        "sun": CapacityFactor(name="sun", profile=insolation_factor),
        "wind": CapacityFactor(name="wind", profile=wind_capacity_factor),
    }


@pytest.fixture
def generator_types(cop: Series) -> dict[str, GeneratorType]:
    return {
        "pp_coal": create_generator_type(
            name="pp_coal",
            fuel="coal",
            energy_types={
                EE,
            },
            efficiency=pd.DataFrame({EE: [0.9] * N_HOURS}),
        ),
        "pp_gas": create_generator_type(
            name="pp_gas",
            fuel="gas",
            energy_types={
                EE,
            },
            efficiency=pd.DataFrame({EE: [0.8] * N_HOURS}),
        ),
        "heat_plant_coal": create_generator_type(
            name="heat_plant_coal",
            fuel="coal",
            energy_types={HEAT},
            efficiency=pd.DataFrame({HEAT: [0.9] * N_HOURS}),
            life_time=30,
            build_time=1,
        ),
        "local_coal_heat_plant": create_generator_type(
            name="local_coal_heat_plant",
            fuel="coal",
            energy_types={HEAT},
            efficiency=pd.DataFrame({HEAT: [0.88] * N_HOURS}),
            life_time=25,
            build_time=1,
        ),
        "local_coal_heat_plant2": create_generator_type(
            name="local_coal_heat_plant2",
            fuel="coal",
            energy_types={HEAT},
            efficiency=pd.DataFrame({HEAT: [0.88] * N_HOURS}),
            life_time=25,
            build_time=1,
        ),
        "heat_plant_biomass": create_generator_type(
            name="heat_plant_biomass",
            fuel="biomass",
            energy_types={HEAT},
            efficiency=pd.DataFrame({HEAT: [0.5] * N_HOURS}),
            life_time=25,
            build_time=1,
        ),
        "chp_coal": create_generator_type(
            name="chp_coal",
            fuel="coal",
            energy_types={HEAT, EE},
            efficiency=pd.DataFrame({HEAT: [0.5] * N_HOURS, EE: [0.3] * N_HOURS}),
            life_time=30,
            build_time=1,
        ),
        "heat_pump": create_generator_type(
            name="heat_pump",
            conversion_rate={EE: cop},
            energy_types={
                HEAT,
            },
            efficiency=pd.DataFrame({HEAT: [0.9] * N_HOURS}),
        ),
        "boiler_coal": create_generator_type(
            name="boiler_coal",
            fuel="coal",
            energy_types={
                HEAT,
            },
            efficiency=pd.DataFrame({HEAT: [0.8] * N_HOURS}),
        ),
        "boiler_biomass": create_generator_type(
            name="boiler_biomass",
            fuel="biomass",
            energy_types={
                HEAT,
            },
            efficiency=pd.DataFrame({HEAT: [0.84] * N_HOURS}),
        ),
        "pv": create_generator_type(
            name="pv",
            energy_types={
                EE,
            },
            capacity_factor="sun",
            efficiency=pd.DataFrame({EE: [0.9] * N_HOURS}),
        ),
        "solar": create_generator_type(
            name="solar",
            energy_types={
                HEAT,
            },
            capacity_factor="sun",
            efficiency=pd.DataFrame({HEAT: [0.8] * N_HOURS}),
        ),
        "wind_farm": create_generator_type(
            name="wind_farm",
            energy_types={
                EE,
            },
            capacity_factor="wind",
            efficiency=pd.DataFrame({EE: [0.85] * N_HOURS}),
        ),
    }


@pytest.fixture
def storage_types() -> dict[str, StorageType]:
    return {
        "heat_storage_type": create_storage_type(
            name="heat_storage_type", energy_type=HEAT
        ),
        "ee_storage_type": create_storage_type(name="ee_storage_type", energy_type=EE),
    }


@pytest.fixture
def generation_fraction() -> GenerationFraction:
    return GenerationFraction(
        name="boiler_coal_solar_generation_fraction",
        tag="example_tag",
        sub_tag="example_sub_tag",
        energy_type=HEAT,
        fraction_type="yearly",
        min_generation_fraction=pd.Series([np.nan, 0, np.nan, np.nan, 0]),
        max_generation_fraction=pd.Series([np.nan, 0.5, np.nan, np.nan, 0.9]),
    )


@pytest.fixture
def empty_network(
    network_constants: NetworkConstants,
    fuels: dict[str, Fuel],
    cfs: dict[str, CapacityFactor],
    generator_types: dict[str, GeneratorType],
    storage_types: dict[str, StorageType],
) -> Network:
    _network = Network(
        energy_types=[HEAT, EE],
        emission_types=[CO2, PM10],
        network_constants=network_constants,
    )
    _network.add_fuel(fuels["coal"])
    _network.add_fuel(fuels["gas"])
    _network.add_fuel(fuels["biomass"])
    _network.add_capacity_factor(cfs["sun"])
    _network.add_capacity_factor(cfs["wind"])

    for generator_type in generator_types.values():
        _network.add_generator_type(generator_type)

    for storage_type in storage_types.values():
        _network.add_storage_type(storage_type)

    return _network
