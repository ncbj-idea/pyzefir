import numpy as np
import pandas as pd
import pytest

from pyzefir.model.network import Network
from pyzefir.model.network_elements import Generator, GeneratorType
from tests.unit.optimization.gurobi.constants import N_YEARS
from tests.unit.optimization.gurobi.test_model.utils import (
    create_default_opf_config,
    run_opt_engine,
)


@pytest.fixture
def network_with_additional_gen(
    network: Network,
    generator_types: dict[str, GeneratorType],
    coal_heat_plant: Generator,
) -> Network:
    network.add_generator_type(generator_types["heat_plant_coal"])
    network.add_generator(coal_heat_plant)
    return network


@pytest.mark.parametrize(
    ("year_sample", "hour_sample", "fuel_availability", "energy_per_unit"),
    [
        (
            np.arange(4),
            np.arange(60),
            {
                "coal": np.array([0.5, 1, 0.8, 0.6, 0.9]),
                "biomass": np.array([100, 120, 130, 140, 150]),
            },
            {"coal": 15, "biomass": 15},
        ),
        (
            np.arange(5),
            np.arange(100),
            {
                "coal": np.array([1.5, 1, 1.8, 1.6, 1.9]),
                "biomass": np.array([100, 120, 130, 140, 150]),
            },
            {"coal": 10, "biomass": 15},
        ),
        (
            np.arange(3),
            np.arange(50),
            {
                "coal": np.array([5, 1, 2, 7, 6]),
                "biomass": np.array([300, 900, 800, 200, 300]),
            },
            {"coal": 1.2, "biomass": 150},
        ),
        (
            np.arange(3),
            np.arange(50),
            {
                "coal": np.array([5, 1, np.nan, 7, 6]),
                "biomass": np.array([300, 900, 800, 200, 300]),
            },
            {"coal": 1.2, "biomass": 150},
        ),
        (
            np.arange(3),
            np.arange(50),
            {
                "coal": np.array([5, np.nan, 2, np.nan, np.nan]),
                "biomass": np.array([300, 900, 800, 200, 300]),
            },
            {"coal": 1.2, "biomass": 150},
        ),
    ],
)
def test_fuel_availability(
    year_sample: np.ndarray,
    hour_sample: np.ndarray,
    network_with_additional_gen: Network,
    fuel_availability: dict[str, np.ndarray],
    energy_per_unit: dict[str, float],
) -> None:
    """
    Conditions to check: generation / energy_per_unit <= fuel_availability
    coal always cheaper than biomass (default settings), so will be built first.
    Intentionally oversized biomass energy_per_unit
    """
    opt_config = create_default_opf_config(hour_sample, year_sample)
    network_with_additional_gen.fuels["coal"].cost = pd.Series(np.zeros(N_YEARS))
    for fuel_name, fuel_aval in fuel_availability.items():
        network_with_additional_gen.fuels[fuel_name].availability = fuel_aval.reshape(
            -1, 1
        )

    for fuel_name, fuel_en_per_unit in energy_per_unit.items():
        network_with_additional_gen.fuels[fuel_name].energy_per_unit = fuel_en_per_unit

    engine = run_opt_engine(network_with_additional_gen, opt_config)

    # total coal usage should be equal to coal availability
    coal_usage = (
        sum(
            [
                engine.results.generators_results.gen[key].sum()
                for key in engine.results.generators_results.gen.keys()
                if "coal" in key
            ]
        )
        / network_with_additional_gen.fuels["coal"].energy_per_unit
    ) * opt_config.hourly_scale
    index_nan = np.argwhere(np.isnan(fuel_availability["coal"][year_sample])).flatten()
    if index_nan:
        fuel_avail = np.delete(fuel_availability["coal"][year_sample], index_nan)
        coal_usage.drop(index_nan, inplace=True)
        assert np.allclose(coal_usage, fuel_avail)

    else:
        assert np.allclose(coal_usage, fuel_availability["coal"][year_sample])
