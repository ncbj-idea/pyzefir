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

import pandas as pd
import pytest
from numpy import all, arange, array, array_equal, linspace, nan, ndarray, ones
from pandas import Series

from pyzefir.model.network import Network, NetworkElementsDict
from pyzefir.model.network_elements import DemandProfile
from pyzefir.optimization.linopy.preprocessing.indices import IndexingSet, Indices
from pyzefir.optimization.linopy.preprocessing.opt_parameters import (
    OptimizationParameters,
)
from pyzefir.optimization.linopy.preprocessing.parameters.aggregated_consumer_parameters import (
    AggregatedConsumerParameters,
)
from pyzefir.optimization.opt_config import OptConfig
from tests.unit.optimization.linopy.preprocessing.utils import (
    aggregated_consumer_factory,
)
from tests.unit.optimization.linopy.utils import compare_vectors_dict


@pytest.mark.parametrize(
    (
        "aggr_names",
        "demand_profile",
        "yearly_energy_use",
        "n_consumers",
        "hour_sample",
        "year_sample",
        "expected_result",
    ),
    [
        (
            ["AGGR"],
            {
                "AGGR": {
                    "ee": Series(arange(100) / arange(100).sum()),
                    "heat": Series(ones(100) / 100),
                }
            },
            {
                "AGGR": {
                    "ee": Series(linspace(1e5, 0.9 * 1e5, 10)),
                    "heat": Series(linspace(1e5, 0.85 * 1e5, 10)),
                }
            },
            pd.Series([1] * 10),
            arange(100),
            arange(10),
            {
                0: {
                    "heat": (ones(100) / 100).reshape(-1, 1)
                    * (linspace(1e5, 0.85 * 1e5, 10)).reshape(1, -1),
                    "ee": (arange(100) / arange(100).sum()).reshape(-1, 1)
                    * linspace(1e5, 0.9 * 1e5, 10).reshape(1, -1),
                }
            },
        ),
        (
            ["AGGR1", "AGGR2"],
            {
                "AGGR1": {
                    "ee": Series([0.1, 0.3, 0.2, 0.15, 0.25]),
                    "heat": Series([0.1, 0.15, 0.2, 0.25, 0.3]),
                },
                "AGGR2": {
                    "ee": Series([0.0, 0.0, 0.2, 0.2, 0.1]),
                    "heat": Series([0.1, 0.2, 0.3, 0.1, 0.3]),
                },
            },
            {
                "AGGR1": {"ee": Series([50, 60, 70]), "heat": Series([55, 80, 95])},
                "AGGR2": {"ee": Series([100, 90, 95]), "heat": Series([70, 80, 90])},
            },
            pd.Series([1] * 3),
            array([0, 2, 3]),
            array([1, 2]),
            {
                0: {
                    "heat": array([0.1, 0.2, 0.25]).reshape(-1, 1)
                    * array([80, 95]).reshape(1, -1),
                    "ee": array([0.1, 0.2, 0.15]).reshape(-1, 1)
                    * array([60, 70]).reshape(1, -1),
                },
                1: {
                    "heat": array([0.1, 0.3, 0.1]).reshape(-1, 1)
                    * array([80, 90]).reshape(1, -1),
                    "ee": array([0.0, 0.2, 0.2]).reshape(-1, 1)
                    * array([90, 95]).reshape(1, -1),
                },
            },
        ),
        ([], {}, {}, pd.Series(), arange(50), arange(10), {}),
    ],
)
def test_get_dem(
    aggr_names: list[str],
    demand_profile: dict[str, dict[str, Series]],
    yearly_energy_use: dict[str, dict[str, Series]],
    n_consumers: pd.Series,
    hour_sample: ndarray,
    year_sample: ndarray,
    expected_result: dict[int, dict[str, ndarray]],
) -> None:
    demand_profiles = NetworkElementsDict(
        {
            name: DemandProfile(name=name, normalized_profile=data)
            for name, data in demand_profile.items()
        }
    )
    aggregated_consumers = NetworkElementsDict(
        {
            name: aggregated_consumer_factory(
                name,
                demand_profile=name,
                yearly_energy_usage=yearly_energy_use[name],
                n_consumers=n_consumers,
            )
            for name in aggr_names
        }
    )

    aggr_idx, hour_idx, year_idx = (
        IndexingSet(array(aggr_names)),
        IndexingSet(hour_sample),
        IndexingSet(year_sample),
    )
    result = AggregatedConsumerParameters.get_dem(
        aggregated_consumers, demand_profiles, aggr_idx, hour_idx, year_idx
    )
    for aggr_id in result:
        assert compare_vectors_dict(result[aggr_id], expected_result[aggr_id])


@pytest.mark.parametrize(
    ("aggr_names", "lbs_names", "base_fractions", "expected_result"),
    [
        (
            ["AGGR1", "AGGR2"],
            ["LBS1", "LBS2", "LBS3", "LBS4", "LBS5"],
            {
                "AGGR1": {"LBS1": 0.7, "LBS2": 0.3},
                "AGGR2": {"LBS3": 0.4, "LBS4": 0.6, "LBS5": 0.0},
            },
            array([[0.7, 0.3, 0.0, 0.0, 0.0], [0.0, 0.0, 0.4, 0.6, 0.0]]),
        ),
        (
            ["AGGR1"],
            ["LBS1", "LBS2", "LBS3"],
            {"AGGR1": {"LBS3": 0.0, "LBS2": 0.0, "LBS1": 1.0}},
            array([[1.0, 0.0, 0.0]]),
        ),
        ([], [], {}, array([])),
    ],
)
def test_get_fr_base(
    aggr_names: list[str],
    lbs_names: list[str],
    base_fractions: dict[str, dict[str, float]],
    expected_result: dict[int, set[int]],
) -> None:
    aggregates = NetworkElementsDict(
        {
            name: aggregated_consumer_factory(name, stack_base_fraction=base_fractions)
            for name, base_fractions in base_fractions.items()
        }
    )

    lbs_idx, aggr_idx = IndexingSet(array(lbs_names)), IndexingSet(array(aggr_names))
    result = AggregatedConsumerParameters.get_fr_base(aggregates, aggr_idx, lbs_idx)

    assert all(result == expected_result)


@pytest.mark.parametrize(
    ("aggr_names", "lbs_names", "base_fractions", "expected_result"),
    [
        (
            ["AGGR1", "AGGR2"],
            ["LBS1", "LBS2", "LBS3", "LBS4", "LBS5"],
            {
                "AGGR1": {"LBS1": 0.7, "LBS2": 0.0},
                "AGGR2": {"LBS3": 0.4, "LBS4": 0.6, "LBS5": 0.0},
            },
            array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]]),
        ),
        (
            ["AGGR1"],
            ["LBS1", "LBS2", "LBS3"],
            {"AGGR1": {"LBS3": 0.0, "LBS2": 0.0, "LBS1": 1.0}},
            array([[1, 1, 1]]),
        ),
        (
            ["AGGR1", "AGGR2", "AGGR3"],
            ["LBS1", "LBS2", "LBS3", "LBS4", "LBS5", "LBS6", "LBS7", "LBS8"],
            {
                "AGGR1": {"LBS2": 0.3, "LBS5": 0.1, "LBS7": 0.6, "LBS3": 0.0},
                "AGGR2": {"LBS1": 0.0, "LBS4": 0.6, "LBS6": 0.4},
                "AGGR3": {"LBS8": 1.0},
            },
            array(
                [
                    [0, 1, 1, 0, 1, 0, 1, 0],
                    [1, 0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                ]
            ),
        ),
        ([], [], {}, array([])),
    ],
)
def test_get_lbs_indicator(
    aggr_names: list[str],
    lbs_names: list[str],
    base_fractions: dict[str, dict[str, float]],
    expected_result: dict[int, set[int]],
) -> None:
    aggregates = NetworkElementsDict(
        {
            name: aggregated_consumer_factory(name, stack_base_fraction=base_fractions)
            for name, base_fractions in base_fractions.items()
        }
    )

    lbs_idx, aggr_idx = IndexingSet(array(lbs_names)), IndexingSet(array(aggr_names))
    result = AggregatedConsumerParameters.get_lbs_indicator(
        aggregates, aggr_idx, lbs_idx
    )

    assert all(result == expected_result)


def test_create(complete_network: Network, opt_config: OptConfig) -> None:
    indices = Indices(complete_network, opt_config)
    result = OptimizationParameters(complete_network, indices, opt_config).aggr

    aggr, demands = (
        complete_network.aggregated_consumers,
        complete_network.demand_profiles,
    )

    assert all(
        result.fr_base
        == AggregatedConsumerParameters.get_fr_base(aggr, indices.AGGR, indices.LBS)
    )
    assert all(
        result.lbs_indicator
        == AggregatedConsumerParameters.get_lbs_indicator(
            aggr, indices.AGGR, indices.LBS
        )
    )

    dem_parameters = AggregatedConsumerParameters.get_dem(
        aggr, demands, indices.AGGR, indices.H, indices.Y
    )
    for aggr_id in indices.AGGR.ord:
        assert compare_vectors_dict(result.dem[aggr_id], dem_parameters[aggr_id])


@pytest.mark.parametrize(
    (
        "aggr_names",
        "lbs_names",
        "fractions",
        "sample",
        "fraction_name",
        "expected_result",
    ),
    [
        pytest.param(
            ["AGGR1", "AGGR2"],
            ["LBS1", "LBS2", "LBS3", "LBS4"],
            {
                "AGGR1": {
                    "LBS1": Series([0.1, 0.2, 0.3]),
                    "LBS2": Series([0.9, 0.8, 0.7]),
                },
                "AGGR2": {
                    "LBS3": Series([0.5, 0.5, 0.5]),
                    "LBS4": Series([0.5, 0.5, 0.5]),
                },
            },
            array([0, 1, 2]),
            "min_fraction",
            {
                0: {0: array([0.1, 0.2, 0.3]), 1: array([0.9, 0.8, 0.7])},
                1: {2: array([0.5, 0.5, 0.5]), 3: array([0.5, 0.5, 0.5])},
            },
            id="2 aggr and 4 lbs",
        ),
        pytest.param(
            ["AGGR1", "AGGR2", "AGGR3"],
            ["LBS1", "LBS2", "LBS3", "LBS4", "LBS5"],
            {
                "AGGR1": {
                    "LBS1": Series([0.1, 0.2, 0.3]),
                    "LBS2": Series([0.9, 0.8, 0.7]),
                    "LBS3": Series([nan, nan, nan]),
                },
                "AGGR2": {"LBS4": Series([1.0, 1.0, 1.0])},
                "AGGR3": {"LBS5": Series([nan, nan, nan])},
            },
            array([0, 1, 2]),
            "max_fraction_decrease",
            {
                0: {
                    0: array([0.1, 0.2, 0.3]),
                    1: array([0.9, 0.8, 0.7]),
                    2: array([nan, nan, nan]),
                },
                1: {3: array([1.0, 1.0, 1.0])},
                2: {4: array([nan, nan, nan])},
            },
            id="3 aggr and 5 lbs",
        ),
        pytest.param([], [], {}, array([]), "max_fraction", {}, id="empty elements"),
    ],
)
def test_get_fraction_assignment(
    aggr_names: list[str],
    lbs_names: list[str],
    fractions: dict[str, dict[str, Series]],
    fraction_name: str,
    sample: ndarray,
    expected_result: dict[int, dict[int, ndarray]],
) -> None:
    aggregated_consumers = NetworkElementsDict(
        {
            name: aggregated_consumer_factory(name, fraction=fraction)
            for name, fraction in fractions.items()
        }
    )

    lbs_idx, aggr_idx = IndexingSet(array(lbs_names)), IndexingSet(array(aggr_names))
    result = AggregatedConsumerParameters.get_fraction_assignment(
        aggregated_consumers=aggregated_consumers,
        aggr_idx=aggr_idx,
        lbs_idx=lbs_idx,
        fraction_attr=fraction_name,
        sample=sample,
    )
    assert all(
        array_equal(result[key1][key2], expected_result[key1][key2])
        for key1 in result
        for key2 in result[key1]
    )
