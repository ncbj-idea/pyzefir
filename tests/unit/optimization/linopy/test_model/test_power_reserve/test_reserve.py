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
from pyzefir.model.utils import NetworkConstants
from pyzefir.utils.functions import invert_dict_of_sets
from tests.unit.optimization.linopy.constants import N_YEARS
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opt_config,
    run_opt_engine,
    set_network_elements_parameters,
)


@pytest.mark.parametrize(
    (
        "gen_tags",
        "power_reserves",
        "hour_sample",
        "stack_base_relative_fraction",
        "n_consumers",
    ),
    [
        (
            {
                "pp_coal_grid": ["tag1"],
                "chp_coal_grid_hs": ["tag1"],
                "local_pv": ["tag2"],
                "local_pv2": ["tag2"],
            },
            {"power_reserves": {"electricity": {"tag1": 10.0, "tag2": 2.0}}},
            np.arange(10),
            0.4,
            pd.Series([15000, 20000, 30000, 40000, 10000]),
        ),
        (
            {
                "pp_coal_grid": ["tag1"],
                "chp_coal_grid_hs": ["tag1"],
                "local_pv": ["tag2"],
                "local_pv2": ["tag2"],
            },
            {"power_reserves": {"electricity": {"tag1": 10.0, "tag2": 14.0}}},
            np.arange(10),
            0.4,
            pd.Series([15000, 20000, 30000, 40000, 10000]),
        ),
        (
            {
                "pp_coal_grid": ["tag1", "tag2"],
                "chp_coal_grid_hs": ["tag1"],
                "local_pv": ["tag2"],
                "local_pv2": ["tag2"],
            },
            {"power_reserves": {"electricity": {"tag1": 10.0, "tag2": 5.0}}},
            np.arange(10),
            0.4,
            pd.Series([15000, 20000, 30000, 40000, 10000]),
        ),
        (
            {
                "pp_coal_grid": ["tag1", "tag2"],
                "chp_coal_grid_hs": ["tag1"],
                "local_pv": ["tag2"],
                "local_pv2": ["tag1", "tag2"],
            },
            {"power_reserves": {"electricity": {"tag1": 10.0, "tag2": 5.0}}},
            np.arange(10),
            0.4,
            pd.Series([15000, 20000, 30000, 40000, 10000]),
        ),
        (
            {"pp_coal_grid": ["tag1"], "local_pv": ["tag1"], "local_pv2": ["tag1"]},
            {"power_reserves": {"electricity": {"tag1": 10.0}}},
            np.arange(10),
            0.5,
            pd.Series([10000, 20000, 30000, 40000, 10000]),
        ),
        (
            {"pp_coal_grid": ["tag1"]},
            {"power_reserves": {"electricity": {"tag1": 10.0}}},
            np.arange(10),
            0.5,
            pd.Series([10000, 20000, 30000, 40000, 10000]),
        ),
        (
            {"pp_coal_grid": ["tag1"]},
            {"power_reserves": {"electricity": {"tag1": 39.9}}},
            np.arange(10),
            0.5,
            pd.Series([1000, 10, 10000, 10, 1000]),
        ),
        (
            {"pp_coal_grid": ["tag1"]},
            {"power_reserves": {"electricity": {"tag1": 30}}},
            np.arange(12),
            0.75,
            pd.Series([1500, 1000, 10, 1500, 1000]),
        ),
        (
            {"pp_coal_grid": ["tag1"]},
            {"power_reserves": {"electricity": {"tag1": 39}}},
            np.arange(10),
            0.6,
            pd.Series([50, 1000, 20, 15, 10]),
        ),
        (
            {"pp_coal_grid": ["tag1"]},
            {"power_reserves": {"heat": {"tag1": 39}}},
            np.arange(10),
            0.6,
            pd.Series([50, 10, 2000, 15, 10]),
        ),
    ],
)
def test_reserves(
    gen_tags: dict[str, list[str]],
    power_reserves: dict[str, dict],
    hour_sample: np.ndarray,
    network: Network,
    stack_base_relative_fraction: float,
    n_consumers: pd.Series,
) -> None:
    """
    test power reserves for various generators in tag
    """

    set_network_elements_parameters(
        network.aggregated_consumers,
        {
            "aggr": {
                "stack_base_fraction": {
                    "lbs": stack_base_relative_fraction,
                    "lbs2": 1 - stack_base_relative_fraction,
                },
                "n_consumers": n_consumers,
            }
        },
    )

    for generator_name, tags in gen_tags.items():
        network.generators[generator_name].tags = tags

    constants = network.constants.__dict__
    network.constants = NetworkConstants(**constants | power_reserves)

    set_network_elements_parameters(
        network.generators,
        {
            "pp_coal_grid": {
                "min_device_nom_power": 5,
            },
            "chp_coal_grid_hs": {"min_device_nom_power": 4, "unit_base_cap": 25},
        },
    ),

    opt_config = create_default_opt_config(hour_sample, np.arange(N_YEARS))
    engine = run_opt_engine(network, opt_config)

    power_reserves = engine.parameters.scenario_parameters.power_reserves
    cap: dict[str, pd.DataFrame] = engine.results.generators_results.cap
    gen_et: dict[str, dict[str, pd.DataFrame]] = (
        engine.results.generators_results.gen_et
    )
    gens_of_tag: dict[int, set[int]] = invert_dict_of_sets(engine.parameters.gen.tags)
    gens_of_tag_str: dict[int, set[str]] = {
        k: {engine.indices.GEN.mapping[gen_idx] for gen_idx in v}
        for k, v in gens_of_tag.items()
    }

    _to_test_reserve(power_reserves, cap, gen_et, gens_of_tag_str)


def _to_test_reserve(
    power_reserves: dict[str, dict[int, float]],
    cap: dict[str, pd.DataFrame],
    gen_et: dict[str, dict[str, pd.DataFrame]],
    gens_of_tag_str: dict[int, set[str]],
) -> None:
    for energy_type, tag_to_reserve in power_reserves.items():
        for tag, reserve in tag_to_reserve.items():
            cap_sum = sum(
                cap[gen_name]["cap"].values for gen_name in gens_of_tag_str[tag]
            )
            gen_sum = sum(
                gen_et[gen_name][energy_type].values
                for gen_name in gens_of_tag_str[tag]
            )
            for h in range(gen_sum.shape[0]):
                assert np.all(cap_sum - gen_sum[h] >= reserve)
