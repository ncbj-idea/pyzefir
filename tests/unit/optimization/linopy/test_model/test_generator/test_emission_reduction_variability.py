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
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.network import Network
from tests.unit.optimization.linopy.constants import N_HOURS, N_YEARS
from tests.unit.optimization.linopy.names import CO2, EE, HEAT, PM10
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opt_config,
    run_opt_engine,
    set_network_elements_parameters,
)


@pytest.mark.parametrize(
    (
        "hour_sample",
        "generator_type_parameters",
        "fuel_parameters",
        "aggregated_consumer_parameters",
        "demand_profile_parameters",
        "expected_emissions",
    ),
    [
        pytest.param(
            np.arange(10),
            {
                "pp_coal": {
                    "emission_reduction": {
                        CO2: pd.Series(np.linspace(0, 1, num=N_YEARS)),
                        PM10: pd.Series(np.ones(N_YEARS)),
                    },
                    "efficiency": pd.DataFrame(
                        columns=[
                            EE,
                        ],
                        data=np.ones(N_HOURS) * 0.8,
                    ),
                },
                "heat_plant_biomass": {
                    "emission_reduction": {
                        PM10: pd.Series(np.linspace(0, 1, num=N_YEARS)),
                        CO2: pd.Series(np.ones(N_YEARS)),
                    },
                    "efficiency": pd.DataFrame(
                        columns=[
                            HEAT,
                        ],
                        data=np.ones(N_HOURS) * 0.7,
                    ),
                },
            },
            {
                "coal": {
                    "emission": {CO2: 1.5, PM10: 1.0},
                    "energy_per_unit": 0.6,
                },
                "biomass": {
                    "emission": {CO2: 1.0, PM10: 0.5},
                    "energy_per_unit": 0.8,
                },
            },
            {
                "aggr": {
                    "yearly_energy_usage": {
                        EE: pd.Series(np.ones(N_YEARS) * 1000),
                        HEAT: pd.Series(np.ones(N_YEARS) * 1000),
                    }
                },
            },
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series(np.ones(N_HOURS) / N_HOURS),
                        EE: pd.Series(np.ones(N_HOURS) / N_HOURS),
                    },
                },
            },
            {
                CO2: 1.5
                * (1 - np.linspace(0, 1, num=N_YEARS))
                * 1000
                * (10 / N_HOURS)
                / 0.6
                / 0.8,
                PM10: 0.5
                * (1 - np.linspace(0, 1, num=N_YEARS))
                * 1000
                * (10 / N_HOURS)
                / 0.8
                / 0.7,
            },
            id="increasing emission reduction",
        ),
        pytest.param(
            np.arange(10),
            {
                "pp_coal": {
                    "emission_reduction": {
                        CO2: pd.Series(np.linspace(1, 0, num=N_YEARS)),
                        PM10: pd.Series(np.ones(N_YEARS)),
                    },
                    "efficiency": pd.DataFrame(
                        columns=[
                            EE,
                        ],
                        data=np.ones(N_HOURS) * 0.2,
                    ),
                },
                "heat_plant_biomass": {
                    "emission_reduction": {
                        PM10: pd.Series(np.linspace(1, 0, num=N_YEARS)),
                        CO2: pd.Series(np.ones(N_YEARS)),
                    },
                    "efficiency": pd.DataFrame(
                        columns=[
                            HEAT,
                        ],
                        data=np.ones(N_HOURS) * 0.5,
                    ),
                },
            },
            {
                "coal": {
                    "emission": {CO2: 1.5, PM10: 1.0},
                    "energy_per_unit": 0.6,
                },
                "biomass": {
                    "emission": {CO2: 1.0, PM10: 0.5},
                    "energy_per_unit": 0.8,
                },
            },
            {
                "aggr": {
                    "yearly_energy_usage": {
                        EE: pd.Series(np.ones(N_YEARS) * 1000),
                        HEAT: pd.Series(np.ones(N_YEARS) * 1000),
                    }
                },
            },
            {
                "multi_family_profile": {
                    "normalized_profile": {
                        HEAT: pd.Series(np.ones(N_HOURS) / N_HOURS),
                        EE: pd.Series(np.ones(N_HOURS) / N_HOURS),
                    },
                },
            },
            {
                CO2: 1.5
                * (1 - np.linspace(1, 0, num=N_YEARS))
                * 1000
                * (10 / N_HOURS)
                / 0.6
                / 0.2,
                PM10: 0.5
                * (1 - np.linspace(1, 0, num=N_YEARS))
                * 1000
                * (10 / N_HOURS)
                / 0.8
                / 0.5,
            },
            id="decreasing emission reduction",
        ),
    ],
)
def test_emission_reduction_variability(
    network: Network,
    hour_sample: np.ndarray,
    generator_type_parameters: dict[str, Any],
    fuel_parameters: dict[str, Any],
    aggregated_consumer_parameters: dict[str, Any],
    demand_profile_parameters: dict[str, Any],
    expected_emissions: dict[str, pd.Series],
) -> None:

    set_network_elements_parameters(network.generator_types, generator_type_parameters)
    set_network_elements_parameters(network.fuels, fuel_parameters)
    set_network_elements_parameters(
        network.aggregated_consumers, aggregated_consumer_parameters
    )
    set_network_elements_parameters(network.demand_profiles, demand_profile_parameters)

    opt_config = create_default_opt_config(
        hour_sample=hour_sample, year_sample=np.arange(N_YEARS)
    )
    engine = run_opt_engine(network, opt_config)

    emissions = dict()
    for generator, generator_parameters in network.generators.items():
        gen_type_data = network.generator_types[generator_parameters.energy_source_type]
        fuel = network.fuels[gen_type_data.fuel]
        emission_reduction = gen_type_data.emission_reduction
        for emission_name, reduction in emission_reduction.items():
            if emission_name not in emissions:
                emissions[emission_name] = np.zeros(N_YEARS)
            fuel_consumption = (
                engine.results.generators_results.gen[generator] / fuel.energy_per_unit
            )
            emissions[emission_name] += (
                (fuel_consumption * fuel.emission[emission_name] * (1 - reduction))
                .sum(axis=0)
                .to_numpy()
            )

    for emission_name, emission_values in expected_emissions.items():
        assert np.allclose(emissions[emission_name], emission_values)
