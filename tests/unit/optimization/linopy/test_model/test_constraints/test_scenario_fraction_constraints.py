"""
1. sprawdzenie czy faktycznie frakcje mieszczą się w limitach min i max i increase/decrease
2. sprawdzenie czy frakcje faktycznie przyrastają stopniowo
"""

from copy import deepcopy
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.network import Network
from pyzefir.model.network_elements import AggregatedConsumer, LocalBalancingStack
from tests.unit.optimization.linopy.constants import N_YEARS
from tests.unit.optimization.linopy.test_model.utils import (
    create_default_opt_config,
    run_opt_engine,
)


@pytest.fixture()
def fraction_network_factory(
    network: Network,
) -> Callable[..., Network]:
    """
    Network for testing fraction change.
    Lbs_grid (connected to grid bus) fraction should increase over years.

    """

    def _create_network(
        min_fraction: pd.Series | None = None,
        max_fraction: pd.Series | None = None,
        max_fraction_increase: pd.Series | None = None,
        max_fraction_decrease: pd.Series | None = None,
    ) -> Network:
        network_ = deepcopy(network)
        network_.add_local_balancing_stack(
            LocalBalancingStack(
                name="lbs_grid",
                buses_out={"electricity": "grid", "heat": "hs"},
                buses={"electricity": {"grid"}, "heat": {"hs"}},
            )
        )
        aggr = network_.aggregated_consumers["aggr"]
        aggr.stack_base_fraction = {"lbs": 0.5, "lbs_grid": 0.5}
        aggr.min_fraction = (
            min_fraction
            if min_fraction is not None
            else {
                "lbs": pd.Series([np.nan, 0, 0, 0, 0]),
                "lbs_grid": pd.Series([np.nan, 0, 0, 0, 0]),
            }
        )
        aggr.max_fraction = (
            max_fraction
            if max_fraction is not None
            else {
                "lbs": pd.Series([np.nan, 1, 1, 1, 1]),
                "lbs_grid": pd.Series([np.nan, 1, 1, 1, 1]),
            }
        )
        aggr.max_fraction_increase = (
            max_fraction_increase
            if max_fraction_increase is not None
            else {
                "lbs": pd.Series([np.nan, 1, 1, 1, 1]),
                "lbs_grid": pd.Series([np.nan, 1, 1, 1, 1]),
            }
        )
        aggr.max_fraction_decrease = (
            max_fraction_decrease
            if max_fraction_decrease is not None
            else {
                "lbs": pd.Series([np.nan, 1, 1, 1, 1]),
                "lbs_grid": pd.Series([np.nan, 1, 1, 1, 1]),
            }
        )
        return network_

    return _create_network


@pytest.mark.parametrize(
    "min_fraction, max_fraction",
    [
        (
            {
                "lbs": pd.Series([np.nan, 0, 1, 0, 1]),
                "lbs_grid": pd.Series([np.nan, 1, 0, 1, 0]),
            },
            {
                "lbs": pd.Series([np.nan, 0, 1, 0, 1]),
                "lbs_grid": pd.Series([np.nan, 1, 0, 1, 0]),
            },
        ),
        (
            {
                "lbs": pd.Series([np.nan, 0.2, 0.5, 0.7, 1]),
                "lbs_grid": pd.Series([np.nan, 0, 0, 0, 0]),
            },
            {
                "lbs": pd.Series([np.nan, 1, 1, 1, 1]),
                "lbs_grid": pd.Series([np.nan, 1, 1, 1, 1]),
            },
        ),
    ],
)
def test_min_max_fraction(
    min_fraction: dict[str, pd.Series],
    max_fraction: dict[str, pd.Series],
    fraction_network_factory: Callable[..., Network],
) -> None:
    """
    Tests if fraction values in each year are within given boundaries
    """
    opt_config = create_default_opt_config(
        hour_sample=np.arange(5),
        year_sample=np.arange(N_YEARS),
    )
    network = fraction_network_factory(
        min_fraction=min_fraction, max_fraction=max_fraction
    )
    engine = run_opt_engine(network, opt_config)
    fraction_result = engine.results.fractions_results.frac
    aggrs: list[AggregatedConsumer] = list(network.aggregated_consumers.values())
    for aggr in aggrs:
        aggr_result = fraction_result[aggr.name]
        for stack in aggr.available_stacks:
            max_fraction = aggr.max_fraction[stack]
            min_fraction = aggr.min_fraction[stack]
            stack_result = aggr_result[stack].iloc[:, 0]
            assert not (min_fraction > stack_result).any()
            assert not (max_fraction < stack_result).any()
        assert np.isclose(
            pd.concat([df for df in aggr_result.values()], axis=1).sum(axis=1), 1
        ).all()


@pytest.mark.parametrize(
    "max_increase, max_decrease",
    [
        (
            {
                "lbs": pd.Series([np.nan, 1, 1, 1, 1]),
                "lbs_grid": pd.Series([np.nan, 0.1, 0.2, 0, 0.1]),
            },
            {
                "lbs": pd.Series([np.nan, 1, 1, 1, 1]),
                "lbs_grid": pd.Series([np.nan, 1, 1, 1, 1]),
            },
        ),
        (
            {
                "lbs": pd.Series([np.nan, 1, 1, 1, 1]),
                "lbs_grid": pd.Series([np.nan, 1, 1, 1, 1]),
            },
            {
                "lbs": pd.Series([np.nan, 0.1, 0.1, 0.1, 0.1]),
                "lbs_grid": pd.Series([np.nan, 1, 1, 1, 1]),
            },
        ),
        (
            {
                "lbs": pd.Series([np.nan, 1, 1, 1, 1]),
                "lbs_grid": pd.Series([np.nan, 1, 1, 1, 1]),
            },
            {
                "lbs": pd.Series([np.nan, 1, 1, 1, 1]),
                "lbs_grid": pd.Series([np.nan, 1, 1, 1, 1]),
            },
        ),
    ],
)
def test_max_fraction_change(
    max_increase: dict[str, pd.Series],
    max_decrease: dict[str, pd.Series],
    fraction_network_factory: Callable[..., Network],
) -> None:
    """
    Tests if fraction values increases as given in max_fraction_increase parameter
    """
    opt_config = create_default_opt_config(
        hour_sample=np.arange(5),
        year_sample=np.arange(N_YEARS),
    )
    network = fraction_network_factory(
        max_fraction_increase=max_increase, max_fraction_decrease=max_decrease
    )
    engine = run_opt_engine(network, opt_config)
    fraction_result = engine.results.fractions_results.frac
    aggrs: list[AggregatedConsumer] = list(network.aggregated_consumers.values())
    for aggr in aggrs:
        aggr_result = fraction_result[aggr.name]
        for stack in aggr.available_stacks:
            fraction_increase = aggr.max_fraction_increase[stack]
            fraction_decrease = aggr.max_fraction_decrease[stack]
            stack_result = aggr_result[stack].iloc[:, 0]
            assert (
                (
                    fraction_increase
                    >= (stack_result.shift(-1) - stack_result)
                    .shift(1)
                    .fillna(0)
                    .round(8)
                )
                | pd.isnull(fraction_increase)
            ).all()
            assert (
                (
                    fraction_decrease
                    >= (stack_result - stack_result.shift(-1)).fillna(0).round(8)
                )
                | pd.isnull(fraction_increase)
            ).all()
        assert np.isclose(
            pd.concat([df for df in aggr_result.values()], axis=1).sum(axis=1), 1
        ).all()
