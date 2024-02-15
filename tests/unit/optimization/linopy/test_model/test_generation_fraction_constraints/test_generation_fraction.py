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
import pytest

from pyzefir.model.network import Network
from pyzefir.model.utils import NetworkConstants
from pyzefir.optimization.linopy.preprocessing.indices import Indices
from pyzefir.optimization.results import GeneratorsResults, StoragesResults
from tests.unit.optimization.linopy.constants import N_YEARS
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opf_config,
    run_opt_engine,
    set_network_elements_parameters,
)
from tests.unit.optimization.linopy.utils import TOL


@pytest.mark.parametrize(
    ("hour_sample", "min_max_gen_frac", "unit_tags"),
    [
        (
            np.arange(50),
            {
                "min_generation_fraction": {
                    "electricity": {("ee_tag", "ee_tag2"): 0.0},
                    "heat": {("heat_tag", "heat_tag2"): 0.0},
                },
                "max_generation_fraction": {
                    "electricity": {("ee_tag", "ee_tag2"): 0.0},
                    "heat": {("heat_tag", "heat_tag2"): 0.0},
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
                "min_generation_fraction": {
                    "electricity": {("ee_tag", "ee_tag2"): 0.0},
                    "heat": {("heat_tag", "heat_tag2"): 0.3},
                },
                "max_generation_fraction": {
                    "electricity": {("ee_tag", "ee_tag2"): 0.0},
                    "heat": {("heat_tag", "heat_tag2"): 0.4},
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
                "min_generation_fraction": {
                    "electricity": {("ee_tag", "ee_tag2"): 0.1},
                    "heat": {("heat_tag", "heat_tag2"): 0.3},
                },
                "max_generation_fraction": {
                    "electricity": {("ee_tag", "ee_tag2"): 0.9},
                    "heat": {("heat_tag", "heat_tag2"): 0.7},
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
            np.arange(100),
            {
                "min_generation_fraction": {
                    "electricity": {("ee_tag", "ee_tag2"): 0.2},
                    "heat": {("heat_tag", "heat_tag2"): 0.5},
                },
                "max_generation_fraction": {
                    "electricity": {("ee_tag", "ee_tag2"): 0.9},
                    "heat": {("heat_tag", "heat_tag2"): 0.8},
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
                "min_generation_fraction": {
                    "electricity": {("ee_tag", "ee_tag2"): 0.5},
                    "heat": {("heat_tag", "heat_tag2"): 0.5},
                },
                "max_generation_fraction": {
                    "electricity": {("ee_tag", "ee_tag2"): 0.5},
                    "heat": {("heat_tag", "heat_tag2"): 0.5},
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
                "min_generation_fraction": {
                    "electricity": {("ee_tag", "ee_tag2"): 0.7},
                    "heat": {("heat_tag", "heat_tag2"): 0.5},
                },
                "max_generation_fraction": {
                    "electricity": {("ee_tag", "ee_tag2"): 0.7},
                    "heat": {("heat_tag", "heat_tag2"): 0.5},
                },
            },
            {
                "pp_coal_grid": ["ee_tag", "ee_tag2"],
                "chp_coal_grid_hs": ["ee_tag", "heat_tag"],
                "coal_heat_plant_hs": ["heat_tag2"],
                "biomass_heat_plant_hs": ["heat_tag"],
            },
        ),
    ],
)
def test_generation_fraction(
    hour_sample: np.ndarray,
    min_max_gen_frac: dict[str, dict[tuple[str, str], float]],
    unit_tags: dict[str, list[str]],
    network: Network,
) -> None:
    """
    Test generation fraction constraints
    """
    constants = network.constants.__dict__
    network.constants = NetworkConstants(**constants | min_max_gen_frac)
    for unit, unit_tag in unit_tags.items():
        set_network_elements_parameters(network.generators, {unit: {"tags": unit_tag}})
    opt_config = create_default_opf_config(hour_sample, np.arange(N_YEARS))
    engine = run_opt_engine(network, opt_config)
    indices = engine.indices
    generators_results = engine.results.generators_results
    storages_results = engine.results.storages_results

    min_fr = engine.parameters.scenario_parameters.min_generation_fraction
    max_fr = engine.parameters.scenario_parameters.max_generation_fraction
    for et, tags_gen_frac in min_fr.items():
        for tags, gen_frac in tags_gen_frac.items():
            # it is sufficient restricting to min in the for, sice the structure of the min_fr and max_fr is similar
            tag, subtag = tags
            min_gen_frac, max_gen_frac = (
                min_fr[et][tags],
                max_fr[et][tags],
            )
            tag_gen_idxs = _unit_of_given_tag(engine.parameters.gen.tags, tag)
            subtag_gen_idxs = _unit_of_given_tag(engine.parameters.gen.tags, subtag)
            tag_stor_idxs = _unit_of_given_tag(engine.parameters.stor.tags, tag)
            subtag_stor_idxs = _unit_of_given_tag(engine.parameters.stor.tags, subtag)

            tag_gen = _expr_gen(
                indices,
                generators_results,
                storages_results,
                et,
                tag_gen_idxs,
                tag_stor_idxs,
            )
            subtag_gen = _expr_gen(
                indices,
                generators_results,
                storages_results,
                et,
                subtag_gen_idxs,
                subtag_stor_idxs,
            )

            assert np.all(subtag_gen + TOL >= tag_gen * min_gen_frac)
            assert np.all(subtag_gen <= tag_gen * max_gen_frac + TOL)


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
) -> np.ndarray:
    gens_et = generators_results.gen_et
    stor = storages_results.gen
    gen_names = {indices.GEN.mapping[gen_idx] for gen_idx in gen_idxs}
    stor_names = {indices.GEN.mapping[stor_idx] for stor_idx in stor_idxs}

    generators_generation_expr: np.ndarray = np.sum(
        [gens_et[gen_name][et] for gen_name in gen_names]
    )
    storages_generation_expr: np.ndarray = np.sum(
        [stor[stor_name] for stor_name in stor_names]
    )

    return generators_generation_expr + storages_generation_expr
