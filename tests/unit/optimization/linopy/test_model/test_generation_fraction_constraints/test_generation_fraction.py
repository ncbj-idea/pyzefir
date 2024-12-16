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

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.network import Network
from pyzefir.model.network_elements import GenerationFraction
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.results import GeneratorsResults, StoragesResults
from tests.unit.optimization.linopy.constants import N_YEARS
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opt_config,
    run_opt_engine,
    set_network_elements_parameters,
)
from tests.unit.optimization.linopy.utils import TOL


@pytest.mark.parametrize(
    ("hour_sample", "generation_fractions", "unit_tags"),
    [
        (
            np.arange(50),
            {
                "ee_frac": {
                    "tag": "ee_tag",
                    "subtag": "ee_tag2",
                    "fraction_type": "hourly",
                    "energy_type": "electricity",
                    "min_generation_fraction": pd.Series([0.1] * 5),
                    "max_generation_fraction": pd.Series([0.9] * 5),
                },
                "heat_frac": {
                    "tag": "heat_tag",
                    "subtag": "heat_tag2",
                    "fraction_type": "hourly",
                    "energy_type": "heat",
                    "min_generation_fraction": pd.Series([0] * 5),
                    "max_generation_fraction": pd.Series([1] * 5),
                },
            },
            {
                "pp_coal_grid": ["ee_tag", "ee_tag2"],
                "chp_coal_grid_hs": ["ee_tag", "heat_tag"],
                "coal_heat_plant_hs": ["heat_tag2"],
                "biomass_heat_plant_hs": ["heat_tag"],
            },
        ),
        (
            np.arange(30),
            {
                "ee_frac": {
                    "tag": "ee_tag",
                    "subtag": "ee_tag2",
                    "fraction_type": "hourly",
                    "energy_type": "electricity",
                    "min_generation_fraction": pd.Series([0.2, 0.3, np.nan, 0.2, 0.3]),
                    "max_generation_fraction": pd.Series([0.9, 1.0, np.nan, 1.0, 0.9]),
                },
                "heat_frac": {
                    "tag": "heat_tag",
                    "subtag": "heat_tag2",
                    "fraction_type": "hourly",
                    "energy_type": "heat",
                    "min_generation_fraction": pd.Series([0.2, 0.3, 0.2, np.nan, 0.3]),
                    "max_generation_fraction": pd.Series([0.9, 1.0, 1.0, np.nan, 0.9]),
                },
            },
            {
                "pp_coal_grid": ["ee_tag", "ee_tag2"],
                "chp_coal_grid_hs": ["ee_tag", "heat_tag"],
                "coal_heat_plant_hs": ["heat_tag2"],
                "biomass_heat_plant_hs": ["heat_tag"],
            },
        ),
        (
            np.arange(50),
            {
                "ee_frac": {
                    "tag": "ee_tag",
                    "subtag": "ee_tag2",
                    "fraction_type": "yearly",
                    "energy_type": "electricity",
                    "min_generation_fraction": pd.Series([0.2, 0.3, np.nan, 0.2, 0.3]),
                    "max_generation_fraction": pd.Series([0.9, 1.0, np.nan, 1.0, 0.9]),
                },
                "heat_frac": {
                    "tag": "heat_tag",
                    "subtag": "heat_tag2",
                    "fraction_type": "yearly",
                    "energy_type": "heat",
                    "min_generation_fraction": pd.Series([0.2, 0.3, 0.2, np.nan, 0.3]),
                    "max_generation_fraction": pd.Series([0.9, 1.0, 1.0, np.nan, 0.9]),
                },
            },
            {
                "pp_coal_grid": ["ee_tag", "ee_tag2"],
                "chp_coal_grid_hs": ["ee_tag", "heat_tag"],
                "coal_heat_plant_hs": ["heat_tag2"],
                "biomass_heat_plant_hs": ["heat_tag"],
            },
        ),
        (
            np.arange(50),
            {
                "ee_frac": {
                    "tag": "ee_tag",
                    "subtag": "ee_tag2",
                    "fraction_type": "yearly",
                    "energy_type": "electricity",
                    "min_generation_fraction": pd.Series(
                        [0.2, 0.3, np.nan, 0.2, np.nan]
                    ),
                    "max_generation_fraction": pd.Series(
                        [0.9, 0.9, np.nan, 0.9, np.nan]
                    ),
                },
                "heat_frac": {
                    "tag": "heat_tag",
                    "subtag": "heat_tag2",
                    "fraction_type": "yearly",
                    "energy_type": "heat",
                    "min_generation_fraction": pd.Series([0.1, 0.2, 0.2, np.nan, 0.3]),
                    "max_generation_fraction": pd.Series(
                        [0.87, 0.89, 0.9, np.nan, 0.9]
                    ),
                },
            },
            {
                "pp_coal_grid": ["ee_tag", "ee_tag2"],
                "ee_storage": ["ee_tag"],
                "chp_coal_grid_hs": ["ee_tag", "heat_tag"],
                "coal_heat_plant_hs": ["heat_tag2"],
                "biomass_heat_plant_hs": ["heat_tag"],
            },
        ),
        (
            np.arange(50),
            {
                "ee_frac": {
                    "tag": "ee_tag",
                    "subtag": "ee_tag2",
                    "fraction_type": "yearly",
                    "energy_type": "electricity",
                    "min_generation_fraction": pd.Series(
                        [0.2, 0.3, np.nan, 0.2, np.nan]
                    ),
                    "max_generation_fraction": pd.Series(
                        [0.9, 0.9, np.nan, 0.9, np.nan]
                    ),
                },
                "heat_frac": {
                    "tag": "heat_tag",
                    "subtag": "heat_tag2",
                    "fraction_type": "hourly",
                    "energy_type": "heat",
                    "min_generation_fraction": pd.Series([0.1, 0.2, 0.2, np.nan, 0.3]),
                    "max_generation_fraction": pd.Series(
                        [0.87, 0.89, 0.9, np.nan, 0.9]
                    ),
                },
            },
            {
                "pp_coal_grid": ["ee_tag", "ee_tag2"],
                "ee_storage": ["ee_tag"],
                "chp_coal_grid_hs": ["ee_tag", "heat_tag"],
                "coal_heat_plant_hs": ["heat_tag2"],
                "biomass_heat_plant_hs": ["heat_tag"],
            },
        ),
        (
            np.arange(50),
            {
                "ee_frac": {
                    "tag": "ee_tag",
                    "subtag": "ee_tag2",
                    "fraction_type": "yearly",
                    "energy_type": "electricity",
                    "min_generation_fraction": pd.Series(
                        [0.2, 0.3, np.nan, 0.2, np.nan]
                    ),
                    "max_generation_fraction": pd.Series(
                        [0.9, 0.9, np.nan, 0.9, np.nan]
                    ),
                },
                "heat_frac": {
                    "tag": "heat_tag",
                    "subtag": "heat_tag2",
                    "fraction_type": "yearly",
                    "energy_type": "heat",
                    "min_generation_fraction": pd.Series([0.1, 0.2, 0.2, np.nan, 0.3]),
                    "max_generation_fraction": pd.Series(
                        [0.87, 0.89, 0.9, np.nan, 0.9]
                    ),
                },
            },
            {
                "pp_coal_grid": ["ee_tag", "ee_tag2"],
                "ee_storage": ["ee_tag"],
                "chp_coal_grid_hs": ["ee_tag", "heat_tag"],
                "coal_heat_plant_hs": ["heat_tag2"],
                "biomass_heat_plant_hs": ["heat_tag"],
            },
        ),
        (
            np.arange(50),
            {
                "ee_frac": {
                    "tag": "ee_tag",
                    "subtag": "ee_tag2",
                    "fraction_type": "yearly",
                    "energy_type": "electricity",
                    "min_generation_fraction": pd.Series([np.nan] * 5),
                    "max_generation_fraction": pd.Series([np.nan] * 5),
                },
                "heat_frac": {
                    "tag": "heat_tag",
                    "subtag": "heat_tag2",
                    "fraction_type": "yearly",
                    "energy_type": "heat",
                    "min_generation_fraction": pd.Series([np.nan] * 5),
                    "max_generation_fraction": pd.Series([np.nan] * 5),
                },
            },
            {
                "pp_coal_grid": ["ee_tag", "ee_tag2"],
                "ee_storage": ["ee_tag"],
                "chp_coal_grid_hs": ["ee_tag", "heat_tag"],
                "coal_heat_plant_hs": ["heat_tag2"],
                "biomass_heat_plant_hs": ["heat_tag"],
            },
        ),
        (
            np.arange(50),
            {
                "ee_frac": {
                    "tag": "ee_tag",
                    "subtag": "ee_tag2",
                    "fraction_type": "hourly",
                    "energy_type": "electricity",
                    "min_generation_fraction": pd.Series([np.nan] * 5),
                    "max_generation_fraction": pd.Series([np.nan] * 5),
                },
                "heat_frac": {
                    "tag": "heat_tag",
                    "subtag": "heat_tag2",
                    "fraction_type": "hourly",
                    "energy_type": "heat",
                    "min_generation_fraction": pd.Series([np.nan] * 5),
                    "max_generation_fraction": pd.Series([np.nan] * 5),
                },
            },
            {
                "pp_coal_grid": ["ee_tag", "ee_tag2"],
                "ee_storage": ["ee_tag"],
                "chp_coal_grid_hs": ["ee_tag", "heat_tag"],
                "coal_heat_plant_hs": ["heat_tag2"],
                "biomass_heat_plant_hs": ["heat_tag"],
            },
        ),
        (
            np.arange(50),
            {
                "ee_frac": {
                    "tag": "ee_tag",
                    "subtag": "ee_tag2",
                    "fraction_type": "hourly",
                    "energy_type": "electricity",
                    "min_generation_fraction": pd.Series([0.2] * 5),
                    "max_generation_fraction": pd.Series([0.5] * 5),
                },
                "heat_frac": {
                    "tag": "heat_tag",
                    "subtag": "heat_tag2",
                    "fraction_type": "hourly",
                    "energy_type": "heat",
                    "min_generation_fraction": pd.Series([0.5] * 5),
                    "max_generation_fraction": pd.Series([0.8, 0.8, 0.7, 0.7, 0.7]),
                },
            },
            {
                "pp_coal_grid": ["ee_tag"],
                "chp_coal_grid_hs": ["ee_tag", "ee_tag2", "heat_tag", "heat_tag2"],
                "coal_heat_plant_hs": ["heat_tag"],
                "biomass_heat_plant_hs": ["heat_tag"],
            },
        ),
        (
            np.arange(50),
            {
                "ee_frac": {
                    "tag": "ee_tag",
                    "subtag": "ee_tag2",
                    "fraction_type": "yearly",
                    "energy_type": "electricity",
                    "min_generation_fraction": pd.Series([0.2] * 5),
                    "max_generation_fraction": pd.Series([0.5] * 5),
                },
                "heat_frac": {
                    "tag": "heat_tag",
                    "subtag": "heat_tag2",
                    "fraction_type": "yearly",
                    "energy_type": "heat",
                    "min_generation_fraction": pd.Series([0.5] * 5),
                    "max_generation_fraction": pd.Series([0.8, 0.8, 0.7, 0.7, 0.7]),
                },
            },
            {
                "pp_coal_grid": ["ee_tag"],
                "chp_coal_grid_hs": ["ee_tag", "ee_tag2", "heat_tag", "heat_tag2"],
                "coal_heat_plant_hs": ["heat_tag"],
                "biomass_heat_plant_hs": ["heat_tag"],
            },
        ),
        (
            np.arange(50),
            {
                "ee_frac": {
                    "tag": "ee_tag",
                    "subtag": "ee_tag2",
                    "fraction_type": "hourly",
                    "energy_type": "electricity",
                    "min_generation_fraction": pd.Series(
                        [0.2, 0.2, np.nan, 0.23, np.nan, np.nan]
                    ),
                    "max_generation_fraction": pd.Series(
                        [0.5, 0.5, np.nan, 0.48, np.nan, np.nan]
                    ),
                },
                "heat_frac": {
                    "tag": "heat_tag",
                    "subtag": "heat_tag2",
                    "fraction_type": "hourly",
                    "energy_type": "heat",
                    "min_generation_fraction": pd.Series(
                        [0.5, np.nan, np.nan, np.nan, 0.5]
                    ),
                    "max_generation_fraction": pd.Series(
                        [0.8, np.nan, np.nan, np.nan, 0.7]
                    ),
                },
            },
            {
                "pp_coal_grid": ["ee_tag", "ee_tag2"],
                "local_pv": ["ee_tag", "ee_tag2"],
                "ee_storage": ["ee_tag", "ee_tag2"],
                "chp_coal_grid_hs": ["ee_tag", "heat_tag"],
                "coal_heat_plant_hs": ["heat_tag2"],
                "biomass_heat_plant_hs": ["heat_tag"],
            },
        ),
        (
            np.arange(50),
            {
                "ee_frac": {
                    "tag": "ee_tag",
                    "subtag": "ee_tag2",
                    "fraction_type": "yearly",
                    "energy_type": "electricity",
                    "min_generation_fraction": pd.Series(
                        [0.2, 0.2, np.nan, 0.23, np.nan, np.nan]
                    ),
                    "max_generation_fraction": pd.Series(
                        [0.5, 0.5, np.nan, 0.48, np.nan, np.nan]
                    ),
                },
                "heat_frac": {
                    "tag": "heat_tag",
                    "subtag": "heat_tag2",
                    "fraction_type": "yearly",
                    "energy_type": "heat",
                    "min_generation_fraction": pd.Series(
                        [0.5, np.nan, np.nan, np.nan, 0.5]
                    ),
                    "max_generation_fraction": pd.Series(
                        [0.8, np.nan, np.nan, np.nan, 0.7]
                    ),
                },
            },
            {
                "pp_coal_grid": ["ee_tag", "ee_tag2"],
                "local_pv": ["ee_tag", "ee_tag2"],
                "ee_storage": ["ee_tag", "ee_tag2"],
                "chp_coal_grid_hs": ["ee_tag", "heat_tag"],
                "coal_heat_plant_hs": ["heat_tag2"],
                "biomass_heat_plant_hs": ["heat_tag"],
            },
        ),
        (
            np.arange(50),
            {
                "ee_frac": {
                    "tag": "ee_tag",
                    "subtag": "ee_tag2",
                    "fraction_type": "hourly",
                    "energy_type": "electricity",
                    "min_generation_fraction": pd.Series(
                        [0.2, 0.2, np.nan, 0.23, np.nan, np.nan]
                    ),
                    "max_generation_fraction": pd.Series(
                        [0.5, 0.5, np.nan, 0.48, np.nan, np.nan]
                    ),
                },
                "heat_frac": {
                    "tag": "heat_tag",
                    "subtag": "heat_tag2",
                    "fraction_type": "yearly",
                    "energy_type": "heat",
                    "min_generation_fraction": pd.Series(
                        [0.5, np.nan, np.nan, np.nan, 0.5]
                    ),
                    "max_generation_fraction": pd.Series(
                        [0.8, np.nan, np.nan, np.nan, 0.7]
                    ),
                },
            },
            {
                "pp_coal_grid": ["ee_tag", "ee_tag2"],
                "local_pv": ["ee_tag", "ee_tag2"],
                "ee_storage": ["ee_tag", "ee_tag2"],
                "chp_coal_grid_hs": ["ee_tag", "heat_tag"],
                "coal_heat_plant_hs": ["heat_tag2"],
                "biomass_heat_plant_hs": ["heat_tag"],
            },
        ),
        (
            np.arange(50),
            {
                "ee_frac": {
                    "tag": "ee_tag",
                    "subtag": "ee_tag2",
                    "fraction_type": "hourly",
                    "energy_type": "electricity",
                    "min_generation_fraction": pd.Series(
                        [0.2, 0.2, np.nan, 0.23, np.nan, np.nan]
                    ),
                    "max_generation_fraction": pd.Series(
                        [0.5, 0.5, np.nan, 0.48, np.nan, np.nan]
                    ),
                },
                "heat_frac": {
                    "tag": "heat_tag",
                    "subtag": "heat_tag2",
                    "fraction_type": "yearly",
                    "energy_type": "heat",
                    "min_generation_fraction": pd.Series(
                        [np.nan, np.nan, np.nan, np.nan, np.nan]
                    ),
                    "max_generation_fraction": pd.Series(
                        [np.nan, np.nan, np.nan, np.nan, np.nan]
                    ),
                },
            },
            {
                "pp_coal_grid": ["ee_tag", "ee_tag2"],
                "local_pv": ["ee_tag", "ee_tag2"],
                "ee_storage": ["ee_tag", "ee_tag2"],
                "chp_coal_grid_hs": ["ee_tag", "heat_tag"],
                "coal_heat_plant_hs": ["heat_tag2"],
                "biomass_heat_plant_hs": ["heat_tag"],
            },
        ),
        (
            np.arange(50),
            {
                "ee_frac": {
                    "tag": "ee_tag",
                    "subtag": "ee_tag2",
                    "fraction_type": "hourly",
                    "energy_type": "electricity",
                    "min_generation_fraction": pd.Series(
                        [np.nan, np.nan, np.nan, np.nan, np.nan]
                    ),
                    "max_generation_fraction": pd.Series([0.5, 0.5, 0.5, 0.48, 0.7]),
                },
                "heat_frac": {
                    "tag": "heat_tag",
                    "subtag": "heat_tag2",
                    "fraction_type": "yearly",
                    "energy_type": "heat",
                    "min_generation_fraction": pd.Series(
                        [np.nan, np.nan, np.nan, np.nan, np.nan]
                    ),
                    "max_generation_fraction": pd.Series(
                        [0.54, 0.51, 0.52, 0.48, 0.48]
                    ),
                },
            },
            {
                "pp_coal_grid": ["ee_tag", "ee_tag2"],
                "local_pv": ["ee_tag", "ee_tag2"],
                "ee_storage": ["ee_tag", "ee_tag2"],
                "chp_coal_grid_hs": ["ee_tag", "heat_tag"],
                "coal_heat_plant_hs": ["heat_tag2"],
                "biomass_heat_plant_hs": ["heat_tag"],
            },
        ),
        (
            np.arange(50),
            {
                "ee_frac": {
                    "tag": "ee_tag",
                    "subtag": "ee_tag2",
                    "fraction_type": "hourly",
                    "energy_type": "electricity",
                    "min_generation_fraction": pd.Series([0.5, 0.5, 0.5, 0.48, 0.7]),
                    "max_generation_fraction": pd.Series(
                        [np.nan, np.nan, np.nan, np.nan, np.nan]
                    ),
                },
                "heat_frac": {
                    "tag": "heat_tag",
                    "subtag": "heat_tag2",
                    "fraction_type": "yearly",
                    "energy_type": "heat",
                    "min_generation_fraction": pd.Series(
                        [0.54, 0.51, 0.52, 0.48, 0.48]
                    ),
                    "max_generation_fraction": pd.Series(
                        [np.nan, np.nan, np.nan, np.nan, np.nan]
                    ),
                },
            },
            {
                "pp_coal_grid": ["ee_tag", "ee_tag2"],
                "local_pv": ["ee_tag", "ee_tag2"],
                "ee_storage": ["ee_tag", "ee_tag2"],
                "chp_coal_grid_hs": ["ee_tag", "heat_tag"],
                "coal_heat_plant_hs": ["heat_tag2"],
                "biomass_heat_plant_hs": ["heat_tag"],
            },
        ),
        (
            np.arange(50),
            {
                "ee_frac": {
                    "tag": "ee_tag",
                    "subtag": "ee_tag2",
                    "fraction_type": "hourly",
                    "energy_type": "electricity",
                    "min_generation_fraction": pd.Series(
                        [0.5, 0.4, 0.6, np.nan, np.nan]
                    ),
                    "max_generation_fraction": pd.Series(
                        [np.nan, np.nan, 0.2, 0.5, 0.7]
                    ),
                },
                "heat_frac": {
                    "tag": "heat_tag",
                    "subtag": "heat_tag2",
                    "fraction_type": "hourly",
                    "energy_type": "heat",
                    "min_generation_fraction": pd.Series(
                        [0.54, np.nan, 0.52, 0.48, np.nan]
                    ),
                    "max_generation_fraction": pd.Series(
                        [0.6, np.nan, np.nan, np.nan, np.nan]
                    ),
                },
            },
            {
                "pp_coal_grid": ["ee_tag", "ee_tag2"],
                "local_pv": ["ee_tag", "ee_tag2"],
                "ee_storage": ["ee_tag", "ee_tag2"],
                "chp_coal_grid_hs": ["ee_tag", "heat_tag"],
                "coal_heat_plant_hs": ["heat_tag2"],
                "biomass_heat_plant_hs": ["heat_tag"],
            },
        ),
        (
            np.arange(50),
            {
                "ee_frac": {
                    "tag": "ee_tag",
                    "subtag": "ee_tag2",
                    "fraction_type": "yearly",
                    "energy_type": "electricity",
                    "min_generation_fraction": pd.Series(
                        [0.5, 0.4, 0.6, np.nan, np.nan]
                    ),
                    "max_generation_fraction": pd.Series(
                        [np.nan, np.nan, 0.2, 0.5, 0.7]
                    ),
                },
                "heat_frac": {
                    "tag": "heat_tag",
                    "subtag": "heat_tag2",
                    "fraction_type": "yearly",
                    "energy_type": "heat",
                    "min_generation_fraction": pd.Series(
                        [0.54, np.nan, 0.52, 0.48, np.nan]
                    ),
                    "max_generation_fraction": pd.Series(
                        [0.6, np.nan, np.nan, np.nan, np.nan]
                    ),
                },
            },
            {
                "pp_coal_grid": ["ee_tag", "ee_tag2"],
                "local_pv": ["ee_tag", "ee_tag2"],
                "ee_storage": ["ee_tag", "ee_tag2"],
                "chp_coal_grid_hs": ["ee_tag", "heat_tag"],
                "coal_heat_plant_hs": ["heat_tag2"],
                "biomass_heat_plant_hs": ["heat_tag"],
            },
        ),
    ],
)
def test_generation_fraction(
    hour_sample: np.ndarray,
    generation_fractions: dict[str, dict],
    unit_tags: dict[str, list[str]],
    network: Network,
) -> None:
    """
    Test generation fraction constraints
    """
    set_network_elements_parameters(
        network.aggregated_consumers,
        {
            "aggr": {
                "n_consumers": pd.Series([1000, 1000, 1000, 1000, 1000]),
            },
        },
    )
    set_network_elements_parameters(
        network.generator_types,
        {
            "pv": {
                "capex": np.array([14, 32, 48, 45, 44]),
                "opex": np.array([23, 12, 48, 45, 44]),
            }
        },
    )
    set_network_elements_parameters(
        network.storage_types,
        {
            "ee_storage_type": {
                "capex": np.array([50, 40, 35, 25, 15]),
                "opex": np.array([40, 12, 20, 22, 10]),
            }
        },
    )

    for fraction_name, generation_fraction in generation_fractions.items():
        network.generation_fractions[fraction_name] = GenerationFraction(
            name=fraction_name,
            tag=generation_fraction["tag"],
            sub_tag=generation_fraction["subtag"],
            fraction_type=generation_fraction["fraction_type"],
            energy_type=generation_fraction["energy_type"],
            min_generation_fraction=generation_fraction["min_generation_fraction"],
            max_generation_fraction=generation_fraction["max_generation_fraction"],
        )

    for unit, unit_tag in unit_tags.items():
        if "ee_storage" not in unit:
            set_network_elements_parameters(
                network.generators, {unit: {"tags": unit_tag}}
            )
        else:
            set_network_elements_parameters(
                network.storages, {unit: {"tags": unit_tag}}
            )

    opt_config = create_default_opt_config(hour_sample, np.arange(N_YEARS))
    engine = run_opt_engine(network, opt_config)
    indices = engine.indices
    generators_results = engine.results.generators_results
    storages_results = engine.results.storages_results
    for idx in engine.indices.GF.ord:
        et = engine.indices.ET.mapping[engine.parameters.gf.et[idx]]
        fraction_type = engine.parameters.gf.fraction_type[idx]
        tag = engine.parameters.gf.tag[idx]
        sub_tag = engine.parameters.gf.sub_tag[idx]
        max_gen_frac = engine.parameters.gf.max_generation_fraction[idx]
        min_gen_frac = engine.parameters.gf.min_generation_fraction[idx]

        max_years = _get_years(engine.indices.Y.ii, max_gen_frac)
        min_years = _get_years(engine.indices.Y.ii, min_gen_frac)

        tag_gen_idxs = _unit_of_given_tag(engine.parameters.gen.tags, tag)
        sub_tag_gen_idxs = _unit_of_given_tag(engine.parameters.gen.tags, sub_tag)
        tag_stor_idxs = _unit_of_given_tag(engine.parameters.stor.tags, tag)
        sub_tag_stor_idxs = _unit_of_given_tag(engine.parameters.stor.tags, sub_tag)

        if len(max_years):
            _check_constraint(
                indices=indices,
                generators_results=generators_results,
                storages_results=storages_results,
                et=et,
                tag_gen_idxs=tag_gen_idxs,
                sub_tag_gen_idxs=sub_tag_gen_idxs,
                tag_stor_idxs=tag_stor_idxs,
                sub_tag_stor_idxs=sub_tag_stor_idxs,
                years=max_years,
                fraction_type=fraction_type,
                network=network,
                frac=max_gen_frac[~np.isnan(max_gen_frac)],
                constraint_type="MAX",
            )
        if len(min_years):
            _check_constraint(
                indices=indices,
                generators_results=generators_results,
                storages_results=storages_results,
                et=et,
                tag_gen_idxs=tag_gen_idxs,
                sub_tag_gen_idxs=sub_tag_gen_idxs,
                tag_stor_idxs=tag_stor_idxs,
                sub_tag_stor_idxs=sub_tag_stor_idxs,
                years=min_years,
                fraction_type=fraction_type,
                network=network,
                frac=min_gen_frac[~np.isnan(min_gen_frac)],
                constraint_type="MIN",
            )


def _check_constraint(
    indices: Indices,
    generators_results: GeneratorsResults,
    storages_results: StoragesResults,
    et: str,
    tag_gen_idxs: set[int],
    sub_tag_gen_idxs: set[int],
    tag_stor_idxs: set[int],
    sub_tag_stor_idxs: set[int],
    years: np.ndarray,
    fraction_type: str,
    network: Network,
    frac: np.ndarray,
    constraint_type: str,
) -> None:
    if len(years):
        tag_gen = _expr_gen(
            indices,
            generators_results,
            storages_results,
            et,
            tag_gen_idxs,
            tag_stor_idxs,
            years=years,
            fraction_type=fraction_type,
            network=network,
        )
        sub_tag_gen = _expr_gen(
            indices,
            generators_results,
            storages_results,
            et,
            sub_tag_gen_idxs,
            sub_tag_stor_idxs,
            years=years,
            fraction_type=fraction_type,
            network=network,
        )
        assert (
            np.all(sub_tag_gen <= tag_gen * frac + TOL)
            if constraint_type == "MAX"
            else np.all(sub_tag_gen + TOL >= tag_gen * frac)
        )


def _get_years(years: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    return np.array([y for y, el in zip(years, fraction) if not np.isnan(el)])


def _unit_of_given_tag(unit_tags: dict[int, set[int]], tag_idx: int) -> set[int]:
    """returns set of units of a given tag"""
    return {gen_idx for gen_idx, tag_set in unit_tags.items() if tag_idx in tag_set}


def _expr_gen(
    indices: Indices,
    generators_results: GeneratorsResults,
    storages_results: StoragesResults,
    et: str,
    gen_idxs: set[int],
    stor_idxs: set[int],
    years: np.ndarray,
    fraction_type: str,
    network: Network,
) -> np.ndarray:
    gens_et = generators_results.gen_et
    stor_gen = storages_results.gen
    gen_names = {indices.GEN.mapping[gen_idx] for gen_idx in gen_idxs}
    stor_names = {indices.STOR.mapping[stor_idx] for stor_idx in stor_idxs}

    zero_mat = np.zeros((len(indices.H.ii), len(years)))
    generators_generation_expr, storages_generation_expr = zero_mat, zero_mat
    if len(gen_idxs):
        generators_generation_expr = sum(
            np.array([gens_et[gen_name][et] for gen_name in gen_names])
        )[:, years]
    if len(stor_names):
        storages_generation_expr = sum(
            np.array([stor_gen[stor_name] for stor_name in stor_names])
        )[:, years]

    if fraction_type == "yearly":
        generators_generation_expr = generators_generation_expr.sum(axis=0)
        storages_generation_expr = storages_generation_expr.sum(axis=0)

    return generators_generation_expr + storages_generation_expr
