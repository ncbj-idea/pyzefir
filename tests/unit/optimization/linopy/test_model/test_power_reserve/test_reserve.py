from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.network import Network
from pyzefir.model.utils import NetworkConstants
from tests.unit.optimization.linopy.constants import N_YEARS
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opt_config,
    run_opt_engine,
    set_network_elements_parameters,
)
from tests.unit.optimization.linopy.utils import TOL


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
                "chp_coal_grid_hs": ["tag2"],
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
                "chp_coal_grid_hs": ["tag2"],
                "local_pv": ["tag2"],
                "local_pv2": ["tag2"],
            },
            {"power_reserves": {"electricity": {"tag1": 4.0, "tag2": 4.0}}},
            np.arange(10),
            0.4,
            pd.Series([1500, 2000, 3000, 4000, 1000]),
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
            {
                "pp_coal_grid": ["tag1", "tag2"],
            },
            {"power_reserves": {"electricity": {"tag1": 10.0, "tag2": 5.0}}},
            np.arange(10),
            0.4,
            pd.Series([15000, 20000, 30000, 40000, 10000]),
        ),
        (
            {
                "pp_coal_grid": ["tag1"],
                "chp_coal_grid_hs": ["tag1", "tag2"],
            },
            {
                "power_reserves": {
                    "electricity": {"tag1": 10.0},
                    "heat": {"tag2": 5.0},
                }
            },
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
            {"power_reserves": {"electricity": {"tag1": 9.9}}},
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
            {"power_reserves": {"electricity": {"tag1": 13}}},
            np.arange(10),
            0.6,
            pd.Series([50, 1000, 20, 15, 10]),
        ),
        (
            {"chp_coal_grid_hs": ["tag1"]},
            {"power_reserves": {"heat": {"tag1": 5}}},
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
    p_res_var: dict[str, dict[str, dict[str, pd.DataFrame]]] = (
        engine.results.generators_results.gen_reserve_et
    )
    p_res = network.constants.power_reserves

    min_p_res_per_tag: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for et in p_res:
        for tag in p_res[et]:
            min_p_res_per_tag[tag][et] += p_res[et][tag]

    for tag_name in min_p_res_per_tag:
        for et in min_p_res_per_tag[tag_name]:
            result = sum(
                p_res_var[tag_name][gen_idx][et] for gen_idx in p_res_var[tag_name]
            ).values
            assert np.all(result >= min_p_res_per_tag[tag_name][et])


@pytest.mark.parametrize(
    (
        "gen_tags",
        "power_reserves",
        "hour_sample",
        "max_capacity",
        "ens_expected",
        "efficiency",
    ),
    [
        pytest.param(
            {
                "pp_coal_grid": ["tag1"],
                "chp_coal_grid_hs": ["tag2"],
                "local_pv": ["tag2"],
                "local_pv2": ["tag2"],
            },
            {"power_reserves": {"electricity": {"tag1": 2.0}}},
            np.arange(10),
            {
                "pv": pd.Series([0] * 5),
                "pp_coal": pd.Series([10] * 5),
                "chp_coal": pd.Series([0] * 5),
            },
            {"local_ee_bus": False},
            {},
            id="small reserve",
        ),
        pytest.param(
            {
                "pp_coal_grid": ["tag1"],
                "chp_coal_grid_hs": ["tag2"],
                "local_pv": ["tag2"],
                "local_pv2": ["tag2"],
            },
            {"power_reserves": {"electricity": {"tag1": 10.0}}},
            np.arange(10),
            {
                "pv": pd.Series([0] * 5),
                "pp_coal": pd.Series([10] * 5),
                "chp_coal": pd.Series([0] * 5),
            },
            {"local_ee_bus": True},
            {"pp_coal": {"electricity": pd.Series([1.0] * 8760)}},
            id="reserve = max capacity",
        ),
        pytest.param(
            {
                "pp_coal_grid": ["tag1"],
                "chp_coal_grid_hs": ["tag2"],
                "local_pv": ["tag2"],
                "local_pv2": ["tag2"],
            },
            {"power_reserves": {"electricity": {"tag1": 7.6}}},
            np.arange(10),
            {
                "pv": pd.Series([1] * 5),
                "pp_coal": pd.Series([10] * 5),
                "chp_coal": pd.Series([0.5] * 5),
            },
            {"local_ee_bus": True},
            {"pp_coal": {"electricity": pd.Series([0.8] * 8760)}},
            id="small reserve, more sources",
        ),
        pytest.param(
            {
                "pp_coal_grid": ["tag1"],
                "chp_coal_grid_hs": ["tag2"],
                "local_pv": ["tag2"],
                "local_pv2": ["tag2"],
            },
            {"power_reserves": {}},
            np.arange(10),
            {
                "pv": pd.Series([0] * 5),
                "pp_coal": pd.Series([10] * 5),
                "chp_coal": pd.Series([0] * 5),
            },
            {"local_ee_bus": False},
            {},
            id="no reserve",
        ),
    ],
)
def test_reserves_by_examining_ens(
    gen_tags: dict[str, list[str]],
    power_reserves: dict[str, dict],
    network: Network,
    hour_sample: np.ndarray,
    max_capacity: dict,
    ens_expected: dict,
    efficiency: dict,
) -> None:
    """
    test power reserve in the following way:
    1. start wit ens-free configuration
    2. set power reserve such that there is not enough power
    3. check if ens is active in result
    """

    set_network_elements_parameters(
        network.aggregated_consumers,
        {
            "aggr": {
                "stack_base_fraction": {
                    "lbs": 0.5,
                    "lbs2": 0.5,
                },
                "n_consumers": pd.Series([10, 12, 15, 20, 30]),
            }
        },
    )

    for generator_type_name, max_capacity_value in max_capacity.items():
        network.generator_types[generator_type_name].max_capacity = max_capacity_value

    for generator_type_name, eff in efficiency.items():
        for et, eff_value in eff.items():
            network.generator_types[generator_type_name].efficiency[et] = eff_value

    for generator_name, tags in gen_tags.items():
        network.generators[generator_name].tags = tags

    constants = network.constants.__dict__
    network.constants = NetworkConstants(**constants | power_reserves)

    network.generators["chp_coal_grid_hs"].unit_base_cap = 0.0
    network.generators["chp_coal_grid_hs"].unit_max_capacity = pd.Series([0] * 5)
    network.generators["pp_coal_grid"].unit_base_cap = 10.0

    opt_config = create_default_opt_config(hour_sample, np.arange(N_YEARS))
    engine = run_opt_engine(network, opt_config)

    for ens_bus, expected_ens_value in ens_expected.items():
        ens_present = {
            ens_bus: np.any(
                np.array(engine.results.bus_results.bus_ens[ens_bus]).sum(axis=0) > TOL
            )
        }
        assert ens_present == ens_expected


def _to_test_reserve(
    power_reserves: dict[str, dict[int, float]],
    gen_reserve_et: dict[str, dict[str, pd.DataFrame]],
    gens_of_tag_str: dict[int, set[str]],
) -> None:
    for energy_type, tag_to_reserve in power_reserves.items():
        for tag, reserve in tag_to_reserve.items():
            gen_sum = sum(
                gen_reserve_et[gen_name][energy_type].values
                for gen_name in gens_of_tag_str[tag]
            )
            for h in range(gen_sum.shape[0]):
                assert np.all(gen_sum + TOL >= reserve)
