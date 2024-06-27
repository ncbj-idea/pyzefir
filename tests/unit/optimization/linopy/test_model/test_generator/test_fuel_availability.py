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

from pyzefir.model.network import Network
from pyzefir.model.network_elements import Generator, GeneratorType
from tests.unit.optimization.linopy.constants import N_YEARS
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opt_config,
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
                "coal": pd.Series([0.5, 1, 0.8, 0.6, 0.9]),
                "biomass": pd.Series([100, 120, 130, 140, 150]),
            },
            {"coal": 15, "biomass": 15},
        ),
        (
            np.arange(5),
            np.arange(100),
            {
                "coal": pd.Series([1.5, 1, 1.8, 1.6, 1.9]),
                "biomass": pd.Series([100, 120, 130, 140, 150]),
            },
            {"coal": 10, "biomass": 15},
        ),
        (
            np.arange(3),
            np.arange(50),
            {
                "coal": pd.Series([5, 1, 2, 7, 6]),
                "biomass": pd.Series([300, 900, 800, 200, 300]),
            },
            {"coal": 1.2, "biomass": 150},
        ),
        (
            np.arange(3),
            np.arange(50),
            {
                "coal": pd.Series([5, 1, np.nan, 7, 6]),
                "biomass": pd.Series([300, 900, 800, 200, 300]),
            },
            {"coal": 1.2, "biomass": 150},
        ),
        (
            np.arange(3),
            np.arange(50),
            {
                "coal": pd.Series([5, np.nan, 2, np.nan, np.nan]),
                "biomass": pd.Series([300, 900, 800, 200, 300]),
            },
            {"coal": 1.2, "biomass": 150},
        ),
        (
            np.arange(3),
            np.arange(50),
            {
                "coal": np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
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
    opt_config = create_default_opt_config(hour_sample, year_sample)
    network_with_additional_gen.fuels["coal"].cost = pd.Series(np.zeros(N_YEARS))
    for fuel_name, fuel_aval in fuel_availability.items():
        network_with_additional_gen.fuels[fuel_name].availability = fuel_aval

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
    if index_nan.any():
        fuel_avail = np.delete(fuel_availability["coal"][year_sample], index_nan)
        coal_usage.drop(index_nan, inplace=True)
        assert np.allclose(coal_usage, fuel_avail)

    else:
        assert np.allclose(coal_usage, fuel_availability["coal"][year_sample])
