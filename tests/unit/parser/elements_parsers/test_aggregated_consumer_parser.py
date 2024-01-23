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

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from pyzefir.model.network_elements import AggregatedConsumer
from pyzefir.parser.elements_parsers.aggregated_consumer_parser import (
    AggregatedConsumerParser,
)
from pyzefir.parser.utils import sanitize_dataset_name
from pyzefir.utils.path_manager import DataCategories, DataSubCategories


@pytest.fixture
def aggregated_consumer_df(csv_root_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_root_path / f"{DataCategories.STRUCTURE}/"
        f"{DataSubCategories.AGGREGATES}.csv"
    )


@pytest.fixture
def stack_df(csv_root_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_root_path / f"{DataCategories.STRUCTURE}/"
        f"{DataSubCategories.TECHNOLOGYSTACK_AGGREGATE}.csv"
    )


@pytest.fixture
def stack_fraction_df(csv_root_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_root_path / f"{DataCategories.INITIAL_STATE}/"
        f"{DataSubCategories.TECHNOLOGYSTACK}.csv"
    )


@pytest.fixture
def yearly_energy_usage_df(csv_root_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_root_path
        / f"{DataCategories.SCENARIO}/scenario_1"
        / f"{DataSubCategories.YEARLY_ENERGY_USAGE}.csv"
    )


@pytest.fixture()
def fraction_df(csv_root_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_root_path
        / f"{DataCategories.SCENARIO}/scenario_1"
        / f"{DataSubCategories.FRACTIONS}.csv"
    )


@pytest.fixture()
def n_consumers_df(csv_root_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_root_path
        / f"{DataCategories.SCENARIO}/scenario_1"
        / f"{DataSubCategories.N_CONSUMERS}.csv"
    )


@pytest.fixture()
def number_of_years() -> int:
    return 4


@pytest.fixture
def aggregated_consumer_parser(
    aggregated_consumer_df: pd.DataFrame,
    stack_df: pd.DataFrame,
    stack_fraction_df: pd.DataFrame,
    yearly_energy_usage_df: pd.DataFrame,
    fraction_df: pd.DataFrame,
    number_of_years: int,
    n_consumers_df: pd.DataFrame,
) -> AggregatedConsumerParser:
    return AggregatedConsumerParser(
        aggregated_consumer_df=aggregated_consumer_df,
        stack_df=stack_df,
        stack_fraction_df=stack_fraction_df,
        yearly_energy_usage_df=yearly_energy_usage_df,
        fraction_df=fraction_df,
        number_of_years=number_of_years,
        n_consumers=n_consumers_df,
    )


def test_aggregated_consumer_parser_init(
    aggregated_consumer_parser: AggregatedConsumerParser,
) -> None:
    assert isinstance(aggregated_consumer_parser.aggregated_consumer_df, pd.DataFrame)
    assert isinstance(aggregated_consumer_parser.stack_df, pd.DataFrame)
    assert isinstance(aggregated_consumer_parser.stack_fraction_df, pd.DataFrame)


def test_create_fractions(
    stack_df: pd.DataFrame, fraction_df: pd.DataFrame, number_of_years: int
) -> None:
    fraction = AggregatedConsumerParser._create_fractions(
        stack_df, fraction_df, number_of_years
    )

    assert isinstance(fraction, dict)
    assert all(
        set(f.keys()) == set(fraction_df["aggregate"].unique())
        for f in fraction.values()
    )
    for fraction_name, fraction_dict in fraction.items():
        for aggregate, tech_stack_dict in fraction_dict.items():
            assert isinstance(tech_stack_dict, dict)
            for tech_stack, fraction_series in tech_stack_dict.items():
                assert (
                    tech_stack
                    in fraction_df[fraction_df["aggregate"] == aggregate][
                        "technology_stack"
                    ].unique()
                )
                assert isinstance(fraction_series, pd.Series)
                assert list(fraction_series.index) == list(range(number_of_years))
                expected_values = (
                    fraction_df[
                        (fraction_df["aggregate"] == aggregate)
                        & (fraction_df["technology_stack"] == tech_stack)
                    ]
                    .set_index("year")[fraction_name]
                    .reindex(range(number_of_years), fill_value=np.nan)
                )
                assert_series_equal(expected_values, fraction_series, check_names=False)


def test_create_stack_base_fractions(
    stack_fraction_df: pd.DataFrame, stack_df: pd.DataFrame
) -> None:
    stack_base_fractions = AggregatedConsumerParser._create_stack_base_fractions(
        stack_fraction_df, stack_df
    )

    assert len(stack_base_fractions) == len(stack_df["aggregate"].unique())
    assert sum(
        len(stack_fractions) for stack_fractions in stack_base_fractions.values()
    ) == len(stack_df)


def test_create_fractions_empty_frame(
    stack_df: pd.DataFrame, number_of_years: int
) -> None:
    empty_fraction_df = pd.DataFrame(
        {"technology_stack": [], "aggregate": [], "year": [], "fraction": []}
    )

    fraction = AggregatedConsumerParser._create_fractions(
        stack_df, empty_fraction_df, number_of_years
    )

    for _, inner_dict in fraction.items():
        for _, series in inner_dict.items():
            assert len(series) == number_of_years
            assert all(pd.isna(series))


def test_aggregated_consumer_create_yearly_energy_usage(
    yearly_energy_usage_df: pd.DataFrame,
) -> None:
    yearly_demand = AggregatedConsumerParser._create_yearly_energy_usage(
        yearly_energy_usage_df
    )
    energy_types = set(yearly_energy_usage_df["energy_type"].values)
    aggregates = set(yearly_energy_usage_df["aggregate"].values)
    assert len(yearly_demand) == len(yearly_energy_usage_df["aggregate"].unique())
    assert set(yearly_demand) == aggregates
    assert all(
        set(yearly_demand[aggregate]) == energy_types for aggregate in yearly_demand
    )
    assert all(
        isinstance(x, pd.Series)
        for aggregate in yearly_demand
        for x in yearly_demand[aggregate].values()
    )


def test_aggregated_consumer_parser_create_aggregated_consumer(
    stack_fraction_df: pd.DataFrame,
    stack_df: pd.DataFrame,
    aggregated_consumer_df: pd.DataFrame,
    yearly_energy_usage_df: pd.DataFrame,
    fraction_df: pd.DataFrame,
    number_of_years: int,
    n_consumers_df: pd.DataFrame,
) -> None:
    row = aggregated_consumer_df.loc[1]
    name = row["name"]
    demand_type = sanitize_dataset_name(row["demand_type"])

    stack_base_fractions = AggregatedConsumerParser._create_stack_base_fractions(
        stack_fraction_df, stack_df
    )
    yearly_energy_usage = AggregatedConsumerParser._create_yearly_energy_usage(
        yearly_energy_usage_df
    )
    fraction = AggregatedConsumerParser._create_fractions(
        stack_df, fraction_df, number_of_years
    )
    n_consumers = AggregatedConsumerParser._create_consumers(
        n_consumers_df, number_of_years
    )

    aggregated_consumer = AggregatedConsumerParser._create_aggregated_consumer(
        row,
        stack_base_fractions,
        yearly_energy_usage,
        fraction,
        n_consumers,
        number_of_years,
    )

    assert isinstance(aggregated_consumer, AggregatedConsumer)
    assert aggregated_consumer.name == name
    assert aggregated_consumer.demand_profile == demand_type
    assert aggregated_consumer.stack_base_fraction == stack_base_fractions[name]
    assert aggregated_consumer.yearly_energy_usage == yearly_energy_usage[name]
    for frac in [
        "min_fraction",
        "max_fraction",
        "max_fraction_increase",
        "max_fraction_decrease",
    ]:
        assert getattr(aggregated_consumer, frac) == fraction[frac][name]


def test_aggregated_consumer_parser_create(
    aggregated_consumer_parser: AggregatedConsumerParser,
    aggregated_consumer_df: pd.DataFrame,
) -> None:
    aggregates = aggregated_consumer_parser.create()
    assert isinstance(aggregates, tuple)
    assert all(isinstance(aggr, AggregatedConsumer) for aggr in aggregates)
    assert len(aggregates) == len(aggregated_consumer_df)
