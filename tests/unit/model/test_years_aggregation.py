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

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.network import Network
from pyzefir.model.network_aggregator import NetworkAggregator, aggregation_schemas
from pyzefir.model.network_aggregator.utils import (
    DataAggregationItem,
    DemandChunkItemWrapper,
    LastAggregationItem,
    MeanAggregationItem,
    SumAggregationItem,
)
from pyzefir.model.network_elements import GeneratorType, StorageType
from pyzefir.model.utils import NetworkConstants
from pyzefir.utils.config_parser import ConfigParams
from tests.unit.defaults import (
    CO2_EMISSION,
    ELECTRICITY,
    HEATING,
    PM10_EMISSION,
    default_network_constants,
)


@pytest.fixture()
def network() -> Network:
    return Network(
        energy_types=[ELECTRICITY, HEATING],
        network_constants=NetworkConstants(
            **(
                default_network_constants.__dict__
                | {
                    "base_total_emission": {},
                    "relative_emission_limits": {},
                }
            )
        ),
        emission_types=[CO2_EMISSION, PM10_EMISSION],
    )


@pytest.fixture(scope="module")
def config_params() -> ConfigParams:
    return ConfigParams(
        input_path=Path(""),
        scenario="",
        input_format="xlsx",
        output_path=Path(""),
        csv_dump_path=Path(""),
        opt_logs_path=Path(""),
        sol_dump_path=Path("res.sol"),
        year_sample=None,
        hour_sample=None,
        discount_rate=None,
        n_hours=None,
        n_years=None,
        log_level=1,
        network_config={
            "binary_fraction": False,
            "ens_penalty_cost": np.nan,
            "generator_capacity_cost": "brutto",
        },
    )


@pytest.mark.parametrize(
    "year_sample, data, n_years_aggregation, excepted_value, method",
    [
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            1,
            [np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "last",
        ),
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [np.nan, 1, 2, 3, 4, 5, 1, 2, 3, 4, 10],
            5,
            [np.nan, 3.0, 4.0],
            "mean",
        ),
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [np.nan, 1, 2, 3, 4, 5, 3, 2, 3, 4, 10],
            6,
            [np.nan, 3, 4.75],
            "combined",
        ),
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [np.nan, 5, 5, 5, 5, 5, 3, 3, 3, 2, 10],
            3,
            [np.nan, 5, 3, 2, 10],
            "last",
        ),
    ],
)
def test_aggregation_agg_method_switch(
    network: Network,
    config_params: ConfigParams,
    year_sample: list[int],
    data: list[float],
    n_years_aggregation: int,
    excepted_value: list[float],
    method: str,
) -> None:
    network.constants = NetworkConstants(
        **(network.constants.__dict__ | {"n_years": len(year_sample)})
    )

    config_params = ConfigParams(
        **(
            config_params.__dict__
            | {
                "year_sample": np.array(year_sample),
                "n_years_aggregation": n_years_aggregation,
            }
        )
    )

    other_params = [
        "min_capacity_increase",
        "max_capacity_increase",
        "min_capacity",
        "max_capacity",
    ]
    params_to_aggregate = ["capex", "opex"]

    network.add_generator_type(
        GeneratorType(
            name="test",
            life_time=20,
            build_time=0,
            efficiency=pd.DataFrame(
                {
                    ELECTRICITY: [0.5] * default_network_constants.n_hours,
                    HEATING: [0.4] * default_network_constants.n_hours,
                }
            ),
            energy_types={ELECTRICITY, HEATING},
            emission_reduction={
                CO2_EMISSION: pd.Series(data=data, index=year_sample),
                PM10_EMISSION: pd.Series(data=data, index=year_sample),
            },
            power_utilization=pd.Series(data=[1.0] * network.constants.n_hours),
            minimal_power_utilization=pd.Series(data=[0.0] * network.constants.n_hours),
            **{
                param: pd.Series(data=data, index=year_sample)
                for param in params_to_aggregate + other_params
            }
        )
    )
    network_aggregator = NetworkAggregator(
        n_years=network.constants.n_years,
        n_years_aggregation=config_params.n_years_aggregation,
        year_sample=config_params.year_sample,
        aggregation_method=method,
    )
    network_aggregator.aggregate_network(network)

    generator_type = network.generator_types.get("test")

    for param in params_to_aggregate:
        assert getattr(generator_type, param).equals(pd.Series(data=excepted_value))

    for emission in [CO2_EMISSION, PM10_EMISSION]:
        assert generator_type.emission_reduction[emission].equals(
            pd.Series(data=excepted_value)
        )


@pytest.mark.parametrize("name", ["last", "mean", "combined"])
def test_for_aggregation_scheme_existence(name: str) -> None:
    assert aggregation_schemas.__dict__.get(name.upper()) is not None


def assert_objects_equal(
    object_1: dict,
    object_2: dict,
) -> None:
    assert object_1.keys() == object_2.keys()

    for key in object_1.keys():
        if isinstance(object_1[key], pd.Series):
            assert object_1[key].equals(object_2[key])
        elif isinstance(object_1[key], list):
            for i in range(len(object_1[key])):
                assert object_1[key][i].equals(object_2[key][i])
        else:
            assert_objects_equal(object_1[key], object_2[key])


@pytest.mark.parametrize(
    "fake_aggregation_schema, fake_object_to_aggregate, n_years, n_years_aggregation, excepted_object",
    [
        (
            [
                SumAggregationItem(
                    [
                        "item_1",
                        DataAggregationItem.ALL_ELEMENTS,
                        "sub_item_1",
                        DataAggregationItem.ALL_ELEMENTS,
                    ],
                ),
                SumAggregationItem(
                    [
                        "item_1",
                        DataAggregationItem.ALL_ELEMENTS,
                        "sub_item_2",
                    ],
                ),
                SumAggregationItem(
                    ["item_2"],
                ),
            ],
            {
                "item_1": {
                    "test_123": {
                        "sub_item_1": {
                            "test_234": pd.Series(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                            "test_432": pd.Series(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                        },
                        "sub_item_2": pd.Series(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    },
                    "test_321": {
                        "sub_item_1": {
                            "test_234": pd.Series(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                            "test_432": pd.Series(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                        },
                        "sub_item_2": pd.Series(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    },
                },
                "item_2": pd.Series(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            },
            10,
            2,
            {
                "item_1": {
                    "test_123": {
                        "sub_item_1": {
                            "test_234": pd.Series(data=[1, 5, 9, 13, 17, 10]),
                            "test_432": pd.Series(data=[1, 5, 9, 13, 17, 10]),
                        },
                        "sub_item_2": pd.Series(data=[1, 5, 9, 13, 17, 10]),
                    },
                    "test_321": {
                        "sub_item_1": {
                            "test_234": pd.Series(data=[1, 5, 9, 13, 17, 10]),
                            "test_432": pd.Series(data=[1, 5, 9, 13, 17, 10]),
                        },
                        "sub_item_2": pd.Series(data=[1, 5, 9, 13, 17, 10]),
                    },
                },
                "item_2": pd.Series(data=[1, 5, 9, 13, 17, 10]),
            },
        ),
        (
            [
                LastAggregationItem(
                    [
                        "item_1",
                        DataAggregationItem.ALL_ELEMENTS,
                        "sub_item_1",
                        DataAggregationItem.ALL_ELEMENTS,
                    ],
                ),
                SumAggregationItem(
                    [
                        "item_1",
                        DataAggregationItem.ALL_ELEMENTS,
                        "sub_item_2",
                    ],
                ),
                MeanAggregationItem(
                    ["item_2"],
                ),
            ],
            {
                "item_1": {
                    "test_123": {
                        "sub_item_1": {
                            "test_234": pd.Series(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                            "test_432": pd.Series(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                        },
                        "sub_item_2": pd.Series(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    },
                    "test_321": {
                        "sub_item_1": {
                            "test_234": pd.Series(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                            "test_432": pd.Series(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                        },
                        "sub_item_2": pd.Series(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                    },
                },
                "item_2": pd.Series(data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            },
            10,
            6,
            {
                "item_1": {
                    "test_123": {
                        "sub_item_1": {
                            "test_234": pd.Series(data=[1, 7, 10]),
                            "test_432": pd.Series(data=[1, 7, 10]),
                        },
                        "sub_item_2": pd.Series(data=[1, 27, 27]),
                    },
                    "test_321": {
                        "sub_item_1": {
                            "test_234": pd.Series(data=[1, 7, 10]),
                            "test_432": pd.Series(data=[1, 7, 10]),
                        },
                        "sub_item_2": pd.Series(data=[1, 27, 27]),
                    },
                },
                "item_2": pd.Series(data=[1, 4.5, 9]),
            },
        ),
        (
            [
                DemandChunkItemWrapper(
                    LastAggregationItem(
                        [
                            "item_1",
                            DataAggregationItem.ALL_ELEMENTS,
                            "sub_item_1",
                            DataAggregationItem.ALL_ELEMENTS,
                        ],
                    )
                ),
                DemandChunkItemWrapper(
                    SumAggregationItem(
                        [
                            "item_1",
                            DataAggregationItem.ALL_ELEMENTS,
                            "sub_item_2",
                        ],
                    )
                ),
                DemandChunkItemWrapper(
                    MeanAggregationItem(
                        ["item_2"],
                    )
                ),
            ],
            {
                "item_1": {
                    "test_123": {
                        "sub_item_1": {
                            "test_234": np.array(
                                [
                                    [
                                        1,
                                        2,
                                        3,
                                        4,
                                        5,
                                        6,
                                        7,
                                        8,
                                        9,
                                        10,
                                        11,
                                        12,
                                        13,
                                        14,
                                        15,
                                        16,
                                        17,
                                        18,
                                        19,
                                        20,
                                        21,
                                        22,
                                        23,
                                        24,
                                        25,
                                        26,
                                        27,
                                        28,
                                        29,
                                        30,
                                        31,
                                    ]
                                ]
                            ),
                            "test_432": np.array(
                                [
                                    [
                                        1,
                                        2,
                                        3,
                                        4,
                                        5,
                                        6,
                                        7,
                                        8,
                                        9,
                                        10,
                                        11,
                                        12,
                                        13,
                                        14,
                                        15,
                                        16,
                                        17,
                                        18,
                                        19,
                                        20,
                                        21,
                                        22,
                                        23,
                                        24,
                                        25,
                                        26,
                                        27,
                                        28,
                                        29,
                                        30,
                                        31,
                                    ]
                                ]
                            ),
                        },
                        "sub_item_2": np.array(
                            [
                                [
                                    1,
                                    2,
                                    3,
                                    4,
                                    5,
                                    6,
                                    7,
                                    8,
                                    9,
                                    10,
                                    11,
                                    12,
                                    13,
                                    14,
                                    15,
                                    16,
                                    17,
                                    18,
                                    19,
                                    20,
                                    21,
                                    22,
                                    23,
                                    24,
                                    25,
                                    26,
                                    27,
                                    28,
                                    29,
                                    30,
                                    31,
                                ],
                                [
                                    1,
                                    2,
                                    3,
                                    4,
                                    5,
                                    6,
                                    7,
                                    8,
                                    9,
                                    10,
                                    11,
                                    12,
                                    13,
                                    14,
                                    15,
                                    16,
                                    17,
                                    18,
                                    19,
                                    20,
                                    21,
                                    22,
                                    23,
                                    24,
                                    25,
                                    26,
                                    27,
                                    28,
                                    29,
                                    30,
                                    31,
                                ],
                            ]
                        ),
                    },
                    "test_321": {
                        "sub_item_1": {
                            "test_234": np.array(
                                [
                                    [
                                        1,
                                        2,
                                        3,
                                        4,
                                        5,
                                        6,
                                        7,
                                        8,
                                        9,
                                        10,
                                        11,
                                        12,
                                        13,
                                        14,
                                        15,
                                        16,
                                        17,
                                        18,
                                        19,
                                        20,
                                        21,
                                        22,
                                        23,
                                        24,
                                        25,
                                        26,
                                        27,
                                        28,
                                        29,
                                        30,
                                        31,
                                    ]
                                ]
                            ),
                            "test_432": np.array(
                                [
                                    [
                                        1,
                                        2,
                                        3,
                                        4,
                                        5,
                                        6,
                                        7,
                                        8,
                                        9,
                                        10,
                                        11,
                                        12,
                                        13,
                                        14,
                                        15,
                                        16,
                                        17,
                                        18,
                                        19,
                                        20,
                                        21,
                                        22,
                                        23,
                                        24,
                                        25,
                                        26,
                                        27,
                                        28,
                                        29,
                                        30,
                                        31,
                                    ]
                                ]
                            ),
                        },
                        "sub_item_2": np.array(
                            [
                                [
                                    1,
                                    2,
                                    3,
                                    4,
                                    5,
                                    6,
                                    7,
                                    8,
                                    9,
                                    10,
                                    11,
                                    12,
                                    13,
                                    14,
                                    15,
                                    16,
                                    17,
                                    18,
                                    19,
                                    20,
                                    21,
                                    22,
                                    23,
                                    24,
                                    25,
                                    26,
                                    27,
                                    28,
                                    29,
                                    30,
                                    31,
                                ],
                                [
                                    1,
                                    2,
                                    3,
                                    4,
                                    5,
                                    6,
                                    7,
                                    8,
                                    9,
                                    10,
                                    11,
                                    12,
                                    13,
                                    14,
                                    15,
                                    16,
                                    17,
                                    18,
                                    19,
                                    20,
                                    21,
                                    22,
                                    23,
                                    24,
                                    25,
                                    26,
                                    27,
                                    28,
                                    29,
                                    30,
                                    31,
                                ],
                                [
                                    1,
                                    2,
                                    3,
                                    4,
                                    5,
                                    6,
                                    7,
                                    8,
                                    9,
                                    10,
                                    11,
                                    12,
                                    13,
                                    14,
                                    15,
                                    16,
                                    17,
                                    18,
                                    19,
                                    20,
                                    21,
                                    22,
                                    23,
                                    24,
                                    25,
                                    26,
                                    27,
                                    28,
                                    29,
                                    30,
                                    31,
                                ],
                                [
                                    1,
                                    2,
                                    3,
                                    4,
                                    5,
                                    6,
                                    7,
                                    8,
                                    9,
                                    10,
                                    11,
                                    12,
                                    13,
                                    14,
                                    15,
                                    16,
                                    17,
                                    18,
                                    19,
                                    20,
                                    21,
                                    22,
                                    23,
                                    24,
                                    25,
                                    26,
                                    27,
                                    28,
                                    29,
                                    30,
                                    31,
                                ],
                            ]
                        ),
                    },
                },
                "item_2": np.array(
                    [
                        [
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                            15,
                            16,
                            17,
                            18,
                            19,
                            20,
                            21,
                            22,
                            23,
                            24,
                            25,
                            26,
                            27,
                            28,
                            29,
                            30,
                            31,
                        ]
                    ]
                ),
            },
            31,
            5,
            {
                "item_1": {
                    "test_123": {
                        "sub_item_1": {
                            "test_234": [pd.Series([1, 6, 11, 16, 21, 26, 31])],
                            "test_432": [pd.Series([1, 6, 11, 16, 21, 26, 31])],
                        },
                        "sub_item_2": [
                            pd.Series([1, 20, 45, 70, 95, 120, 145]),
                            pd.Series([1, 20, 45, 70, 95, 120, 145]),
                        ],
                    },
                    "test_321": {
                        "sub_item_1": {
                            "test_234": [pd.Series([1, 6, 11, 16, 21, 26, 31])],
                            "test_432": [pd.Series([1, 6, 11, 16, 21, 26, 31])],
                        },
                        "sub_item_2": [
                            pd.Series([1, 20, 45, 70, 95, 120, 145]),
                            pd.Series([1, 20, 45, 70, 95, 120, 145]),
                            pd.Series([1, 20, 45, 70, 95, 120, 145]),
                            pd.Series([1, 20, 45, 70, 95, 120, 145]),
                        ],
                    },
                },
                "item_2": [pd.Series([1.0, 4, 9, 14, 19, 24, 29])],
            },
        ),
    ],
)
def test_fake_aggregation_schema(
    fake_aggregation_schema: list[DataAggregationItem],
    fake_object_to_aggregate: dict[str, pd.Series],
    n_years: int,
    n_years_aggregation: int,
    excepted_object: dict[str, pd.Series],
) -> None:
    network_aggregator = NetworkAggregator(
        n_years=n_years,
        n_years_aggregation=n_years_aggregation,
    )
    fake_network = type(
        "Network",
        (),
        fake_object_to_aggregate | {"constants": default_network_constants},
    )
    network_aggregator._aggregation_scheme = fake_aggregation_schema

    network_aggregator.aggregate_network(fake_network)  # type: ignore

    res = {key: getattr(fake_network, key) for key in fake_object_to_aggregate.keys()}

    assert_objects_equal(res, excepted_object)


@pytest.mark.parametrize(
    "build_time, life_time, n_years_aggregation, excepted_build_time, excepted_life_time",
    [
        (5, 20, 5, 1, 4),
        (4, 20, 1, 4, 20),
        (12, 20, 10, 1, 2),
        (9, 20, 2, 4, 10),
    ],
)
def test_if_build_time_and_lifetime_aggregates_correctly(
    build_time: int,
    life_time: int,
    n_years_aggregation: int,
    excepted_build_time: int,
    excepted_life_time: int,
    network: Network,
) -> None:
    network.add_generator_type(
        GeneratorType(
            name="test",
            life_time=life_time,
            build_time=build_time,
            efficiency=pd.DataFrame(
                {
                    ELECTRICITY: [0.5] * default_network_constants.n_hours,
                    HEATING: [0.4] * default_network_constants.n_hours,
                }
            ),
            energy_types={ELECTRICITY, HEATING},
            emission_reduction={
                CO2_EMISSION: pd.Series(data=[1] * network.constants.n_years),
                PM10_EMISSION: pd.Series(data=[1] * network.constants.n_years),
            },
            power_utilization=pd.Series(data=[1.0] * network.constants.n_hours),
            minimal_power_utilization=pd.Series(data=[0.0] * network.constants.n_hours),
            capex=pd.Series(data=[1] * network.constants.n_years),
            opex=pd.Series(data=[1] * network.constants.n_years),
            min_capacity_increase=pd.Series(data=[np.nan] * network.constants.n_years),
            max_capacity_increase=pd.Series(data=[np.nan] * network.constants.n_years),
            min_capacity=pd.Series(data=[np.nan] * network.constants.n_years),
            max_capacity=pd.Series(data=[np.nan] * network.constants.n_years),
        )
    )

    network.add_storage_type(
        StorageType(
            name="test",
            life_time=life_time,
            build_time=build_time,
            capex=pd.Series(data=[1] * network.constants.n_years),
            opex=pd.Series(data=[1] * network.constants.n_years),
            min_capacity_increase=pd.Series(data=[np.nan] * network.constants.n_years),
            max_capacity_increase=pd.Series(data=[np.nan] * network.constants.n_years),
            min_capacity=pd.Series(data=[np.nan] * network.constants.n_years),
            max_capacity=pd.Series(data=[np.nan] * network.constants.n_years),
            generation_efficiency=0.2,
            load_efficiency=0.5,
            energy_type=ELECTRICITY,
            cycle_length=24,
            power_to_capacity=1.0,
            power_utilization=0.9,
            energy_loss=0.02,
            tags=[],
        )
    )

    network_aggregator = NetworkAggregator(
        n_years=network.constants.n_years,
        n_years_aggregation=n_years_aggregation,
    )

    network_aggregator.aggregate_network(network)

    generator_type = network.generator_types.get("test")

    assert generator_type.build_time == excepted_build_time
    assert generator_type.life_time == excepted_life_time

    storage_type = network.storage_types.get("test")

    assert storage_type.build_time == excepted_build_time
    assert storage_type.life_time == excepted_life_time
