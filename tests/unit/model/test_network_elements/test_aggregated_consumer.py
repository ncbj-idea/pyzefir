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

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.exceptions import (
    NetworkValidatorException,
    NetworkValidatorExceptionGroup,
)
from pyzefir.model.network import Network
from pyzefir.model.network_elements import (
    AggregatedConsumer,
    Bus,
    DemandProfile,
    LocalBalancingStack,
)
from pyzefir.model.utils import NetworkConstants
from tests.unit.defaults import (
    CO2_EMISSION,
    ELECTRICITY,
    HEATING,
    PM10_EMISSION,
    default_energy_profile,
    default_network_constants,
)
from tests.unit.model.test_network_elements.helpers import assert_same_exception_list


@pytest.fixture()
def network() -> Network:
    network = Network(
        emission_types=[CO2_EMISSION, PM10_EMISSION],
        energy_types=[ELECTRICITY, HEATING],
        network_constants=default_network_constants,
    )
    network.add_bus(Bus(name="bus_A", energy_type=ELECTRICITY))
    network.add_bus(Bus(name="bus_B", energy_type=ELECTRICITY))
    network.add_bus(Bus(name="bus_C", energy_type=HEATING))

    network.add_local_balancing_stack(
        LocalBalancingStack(
            name="stack_1",
            buses_out={ELECTRICITY: "bus_A", HEATING: "bus_C"},
            buses={ELECTRICITY: {"bus_A", "bus_B"}, HEATING: {"bus_C"}},
        )
    )
    network.add_local_balancing_stack(
        LocalBalancingStack(
            name="stack_2",
            buses_out={ELECTRICITY: "bus_B", HEATING: "bus_C"},
            buses={ELECTRICITY: {"bus_A", "bus_B"}, HEATING: {"bus_C"}},
        )
    )
    assert len(network.aggregated_consumers) == 0
    demand_profile = DemandProfile("default", default_energy_profile())
    network.add_demand_profile(demand_profile)

    return network


@pytest.fixture()
def aggregate_consumer() -> AggregatedConsumer:
    return AggregatedConsumer(
        name="aggregate_1",
        demand_profile="default",
        stack_base_fraction={"stack_1": 0.5, "stack_2": 0.5},
        yearly_energy_usage={
            ELECTRICITY: pd.Series(data=range(5)),
            HEATING: pd.Series(data=range(5)),
        },
        min_fraction={
            "stack_1": pd.Series([np.nan] * default_network_constants.n_years),
            "stack_2": pd.Series([np.nan] * default_network_constants.n_years),
        },
        max_fraction={
            "stack_1": pd.Series([np.nan] * default_network_constants.n_years),
            "stack_2": pd.Series([np.nan] * default_network_constants.n_years),
        },
        max_fraction_decrease={
            "stack_1": pd.Series([np.nan] * default_network_constants.n_years),
            "stack_2": pd.Series([np.nan] * default_network_constants.n_years),
        },
        max_fraction_increase={
            "stack_1": pd.Series([np.nan] * default_network_constants.n_years),
            "stack_2": pd.Series([np.nan] * default_network_constants.n_years),
        },
        n_consumers=pd.Series([1000] * default_network_constants.n_years),
        average_area=None,
    )


def test_aggregate_setup(aggregate_consumer: AggregatedConsumer) -> None:
    assert aggregate_consumer.name == "aggregate_1"
    assert len(aggregate_consumer.stack_base_fraction) == 2
    assert sum(aggregate_consumer.stack_base_fraction.values()) == 1
    assert isinstance(aggregate_consumer.demand_profile, str)
    assert np.all(
        aggregate_consumer.yearly_energy_usage[ELECTRICITY] == pd.Series(data=range(5))
    )
    assert np.all(
        aggregate_consumer.yearly_energy_usage[HEATING] == pd.Series(data=range(5))
    )
    assert aggregate_consumer.available_stacks == ["stack_1", "stack_2"]
    assert aggregate_consumer.n_consumers.equals(
        pd.Series([1000] * default_network_constants.n_years)
    )


def test_if_all_validators_called(
    network: Network,
    aggregate_consumer: AggregatedConsumer,
) -> None:
    aggregate_consumer._validate_demand_profile = MagicMock()
    aggregate_consumer._validate_stack_base_fraction = MagicMock()
    aggregate_consumer._validate_yearly_energy_usage = MagicMock()
    aggregate_consumer._validate_fractions = MagicMock()
    aggregate_consumer._validate_name_type = MagicMock()
    aggregate_consumer._validate_n_consumers = MagicMock()

    aggregate_consumer.validate(network=network)

    aggregate_consumer._validate_demand_profile.assert_called_once_with(
        network=network, exception_list=[]
    )
    aggregate_consumer._validate_stack_base_fraction.assert_called_once_with(
        network=network, exception_list=[]
    )
    aggregate_consumer._validate_yearly_energy_usage.assert_called_once_with(
        network=network, exception_list=[]
    )
    aggregate_consumer._validate_fractions.assert_called_once_with(
        network=network, exception_list=[]
    )
    aggregate_consumer._validate_name_type.assert_called_once_with(
        exception_list=[],
    )
    aggregate_consumer._validate_n_consumers.assert_called_once_with(
        n_years=network.constants.n_years, exception_list=[]
    )


def test_aggregate_validation_should_rise_exception(
    network: Network,
    aggregate_consumer: AggregatedConsumer,
) -> None:
    aggregate_consumer._validate_demand_profile = MagicMock()
    aggregate_consumer._validate_yearly_energy_usage = MagicMock()
    aggregate_consumer._validate_stack_base_fraction_type = lambda x: x.append(
        NetworkValidatorException("test")
    )

    with pytest.raises(NetworkValidatorException):
        aggregate_consumer.validate(network=network)


@pytest.mark.parametrize(
    "param_name, param_value, exception_list",
    [
        ("demand_profile", "default", []),
        (
            "demand_profile",
            None,
            [
                NetworkValidatorException(
                    "Demand for AggregatedConsumer aggregate_1 must be given as a string"
                )
            ],
        ),
        (
            "demand_profile",
            123,
            [
                NetworkValidatorException(
                    "Demand for AggregatedConsumer aggregate_1 must be given as a string"
                )
            ],
        ),
        (
            "stack_base_fraction",
            {"stack_1": 0.5, "stack_2": 0.2, "stack_3": 0.3},
            [
                NetworkValidatorException(
                    (
                        "Local balancing stack stack_3 available for aggregated "
                        "consumer aggregate_1 does not exist in the network"
                    ),
                )
            ],
        ),
        (
            "stack_base_fraction",
            {"stack_1": 0.5, "stack_2": 0.2},
            [
                NetworkValidatorException(
                    "Local balancing stack fractions for aggregated consumer "
                    "aggregate_1 do not sum to 1, but to 0.7 instead"
                )
            ],
        ),
        (
            "stack_base_fraction",
            {"stack_1": 2, "stack_2": -2},
            [
                NetworkValidatorException(
                    "Local balancing stack fractions for aggregated consumer "
                    "aggregate_1 do not sum to 1, but to 0 instead"
                ),
                NetworkValidatorException(
                    "Fraction of local balancing stack stack_1 in aggregated "
                    "consumer aggregate_1 must be a number from [0,1] interval,"
                    " but 2 given instead"
                ),
                NetworkValidatorException(
                    "Fraction of local balancing stack stack_2 in aggregated "
                    "consumer aggregate_1 must be a number from [0,1] interval,"
                    " but -2 given instead"
                ),
            ],
        ),
        (
            "stack_base_fraction",
            [{"stack_1": 2}],
            [
                NetworkValidatorException(
                    "Stack base fractions for AggregatedConsumer aggregate_1 must be given as a dictionary"
                ),
            ],
        ),
        (
            "stack_base_fraction",
            {"stack_1": "2", "stack_2": -2},
            [
                NetworkValidatorException(
                    "Stack base fractions for AggregatedConsumer aggregate_1 "
                    "must be given as a dictionary with values of type float"
                ),
            ],
        ),
        (
            "stack_base_fraction",
            {12: 2, "stack_2": -2},
            [
                NetworkValidatorException(
                    "Stack base fractions for AggregatedConsumer aggregate_1 "
                    "must be given as a dictionary with keys of type str"
                ),
            ],
        ),
        (
            "stack_base_fraction",
            {1: "stack_1", 2: "stack_2"},
            [
                NetworkValidatorException(
                    "Stack base fractions for AggregatedConsumer aggregate_1 "
                    "must be given as a dictionary with keys of type str"
                ),
                NetworkValidatorException(
                    "Stack base fractions for AggregatedConsumer aggregate_1 "
                    "must be given as a dictionary with values of type float"
                ),
            ],
        ),
    ],
)
def test_aggregate_element_structure_and_fractions_validation(
    network: Network,
    aggregate_consumer: AggregatedConsumer,
    param_name: str,
    param_value: Any,
    exception_list: list,
) -> None:
    actual_exception_list: list[NetworkValidatorException] = []
    if param_name == "demand_profile":
        aggregate_consumer.demand_profile = param_value
        aggregate_consumer._validate_demand_profile(
            network=network, exception_list=actual_exception_list
        )
    elif param_name == "stack_base_fraction":
        aggregate_consumer.stack_base_fraction = param_value
        aggregate_consumer._validate_stack_base_fraction(
            network=network, exception_list=actual_exception_list
        )

    assert_same_exception_list(actual_exception_list, exception_list)


@pytest.mark.parametrize(
    "yearly_energy_usage, exception_list",
    [
        (
            {
                ELECTRICITY: pd.Series(data=range(5)),
                HEATING: pd.Series(data=range(5)),
            },
            [],
        ),
        (
            {12: pd.Series(data=range(5)), HEATING: pd.Series(data=range(5))},
            [NetworkValidatorException("Energy type must be of str type")],
        ),
        (
            {
                type("Object", (), {"x": 5}): "BAD TYPE SERIES",
                HEATING: pd.Series(data=range(5)),
            },
            [
                NetworkValidatorException("Energy usage must be of pd.Series type"),
                NetworkValidatorException("Energy type must be of str type"),
            ],
        ),
        (
            {ELECTRICITY: list(range(5)), HEATING: pd.Series(data=range(5))},
            [
                NetworkValidatorException("Energy usage must be of pd.Series type"),
            ],
        ),
        (
            "BAD DICT",
            [
                NetworkValidatorException("Yearly energy usage must be of dict type"),
            ],
        ),
        (
            ["BAD", "DICT"],
            [
                NetworkValidatorException("Yearly energy usage must be of dict type"),
            ],
        ),
        (
            {
                ELECTRICITY: pd.Series(data=range(5)),
                HEATING: pd.Series(data=range(5)),
            },
            [],
        ),
        (
            {HEATING: pd.Series(data=range(5))},
            [
                NetworkValidatorException(
                    "Energy types of aggregated consumer aggregate_1 are different than "
                    "energy types defined in the connected stack stack_1. Difference: {'ELECTRICITY'}"
                ),
                NetworkValidatorException(
                    "Energy types of aggregated consumer aggregate_1 are different than "
                    "energy types defined in the connected stack stack_2. Difference: {'ELECTRICITY'}"
                ),
            ],
        ),
        (
            {
                ELECTRICITY: pd.Series(data=range(5)),
                HEATING: pd.Series(data=range(6)),
            },
            [
                NetworkValidatorException(
                    "Yearly energy usage series for aggregated consumer aggregate_1 must have the same index"
                )
            ],
        ),
        (
            {
                ELECTRICITY: pd.Series(data=range(5), index=range(2, 7)),
                HEATING: pd.Series(data=range(5)),
            },
            [
                NetworkValidatorException(
                    "Yearly energy usage series for aggregated consumer aggregate_1 must have the same index"
                )
            ],
        ),
        (
            {},
            [
                NetworkValidatorException(
                    "Yearly energy usage must contain at least one energy type"
                )
            ],
        ),
    ],
)
def test_aggregate_yearly_energy_usage(
    network: Network,
    aggregate_consumer: AggregatedConsumer,
    yearly_energy_usage: dict,
    exception_list: list,
) -> None:
    aggregate_consumer.yearly_energy_usage = yearly_energy_usage

    actual_exception_list: list[NetworkValidatorException] = []
    aggregate_consumer._validate_yearly_energy_usage(
        network=network, exception_list=actual_exception_list
    )

    assert_same_exception_list(actual_exception_list, exception_list)


@pytest.mark.parametrize(
    "fraction, exception_list",
    [
        pytest.param(
            {
                "stack_1": pd.Series([np.nan, 0.5, 0.5, 0.5], index=range(4)),
                "stack_2": pd.Series([np.nan, 0.5, 0.5, 0.5], index=range(4)),
            },
            [],
            id="no_errors_no_nan",
        ),
        pytest.param(
            {
                "stack_1": pd.Series([np.nan] * 4, index=range(4)),
                "stack_2": pd.Series([np.nan] * 4, index=range(4)),
            },
            [],
            id="no_errors_with_nan",
        ),
        pytest.param(
            [
                {"stack_1", "stack_2"},
            ],
            [
                NetworkValidatorException(
                    "Fraction min_fraction for aggregate_1 must be of dict type"
                ),
            ],
            id="Fraction list not dict type",
        ),
        pytest.param(
            {
                1: pd.Series([0.5, 0.5, 0.5, 0.5], index=range(4)),
                2: pd.Series([0.5, 0.5, 0.5, 0.5], index=range(4)),
            },
            [
                NetworkValidatorException(
                    "Aggregate in fraction min_fraction for aggregate_1 must be of <class 'str'> type"
                ),
            ],
            id="Dict keys are not str",
        ),
        pytest.param(
            {
                "stack_1": pd.Series([0.5, 0.5, 0.5, 0.5], index=range(4)),
                "stack_2": [[0.5, 0.5, 0.5, 0.5], [0, 1, 2, 3]],
            },
            [
                NetworkValidatorException(
                    "Fraction_series in fraction min_fraction for aggregate_1 must be "
                    "of <class 'pandas.core.series.Series'> type"
                ),
            ],
            id="Value of dict not pd.Series",
        ),
        pytest.param(
            {
                "stack_1": pd.Series([0.5, 0.5, 0.5, 0.5], index=range(4)),
                "stack_2": pd.Series([0.5, 0.5, 0.5, 0.5], index=["1", "2", "3", "4"]),
            },
            [
                NetworkValidatorException(
                    "Local balancing stack stack_2 fraction min_fraction series index "
                    "type is object but should be integer"
                ),
            ],
            id="Series for stack_2 invalid index type ",
        ),
        pytest.param(
            {
                "stack_1": pd.Series(["0.5", 0.5, True, 0.5], index=range(4)),
                "stack_2": pd.Series([np.nan, 0.5, np.nan, 0.5], index=range(4)),
            },
            [
                NetworkValidatorException(
                    "Local balancing stack stack_1 fraction min_fraction series must have only numeric values"
                ),
                NetworkValidatorException(
                    "Local balancing stack stack_1 fraction min_fraction series type is object but should be floating"
                ),
            ],
            id="Series for stack_1 invalid values types ",
        ),
        pytest.param(
            {
                "stack_1": pd.Series([0.1, -0.5, 1.0, 1.23], index=range(4)),
                "stack_2": pd.Series([0.9, 0.5, np.nan, 0.1], index=range(4)),
            },
            [
                NetworkValidatorException(
                    "Fraction min_fraction in LBS stack_1 for AggregatedConsumer "
                    "aggregate_1 values must be given in range <0:1> but"
                    " [-0.5, 1.23] given instead"
                ),
            ],
            id="Series for stack_1 values not in range <0,1>",
        ),
        pytest.param(
            {
                "stack_1": pd.Series([np.nan, np.nan, 0.2, 0.1], index=range(4)),
                "stack_2": pd.Series([np.nan, 0.9, 0.7, 0.1, 1.0, 1.0], index=range(6)),
            },
            [
                NetworkValidatorException(
                    "Local balancing stack stack_2 fraction min_fraction series must have 4 values"
                ),
            ],
            id="stack_2 series range is different than n_years",
        ),
        pytest.param(
            {
                "stack_1": pd.Series([np.nan, np.nan, 0.2, 0.1], index=range(4)),
                "stack_2": pd.Series([0.5, 0.9, 0.7, 0.1], index=range(4)),
            },
            [
                NetworkValidatorException(
                    "Local balancing stack stack_2 in fraction attribute min_fraction "
                    "detected value for base year. This attribute could be provided "
                    "for every year except the base year."
                ),
            ],
            id="fraction values for baseline year not nan",
        ),
        pytest.param(
            {
                "stack_404": pd.Series([np.nan, np.nan, 0.2, 0.1], index=range(4)),
            },
            [
                NetworkValidatorException(
                    "Local balancing stack stack_404 given for fraction min_fraction "
                    "is not defined in base_fractions"
                ),
            ],
            id="lbs not in available stacks",
        ),
    ],
)
def test_validation_aggregate_fraction(
    aggregate_consumer: AggregatedConsumer,
    network: Network,
    fraction: dict | list,
    exception_list: list[NetworkValidatorException],
    fraction_name: str = "min_fraction",
) -> None:
    setattr(aggregate_consumer, fraction_name, fraction)

    actual_exception_list: list[NetworkValidatorException] = []
    aggregate_consumer._validate_fraction(
        fraction_name=fraction_name,
        network=network,
        exception_list=actual_exception_list,
    )

    assert_same_exception_list(actual_exception_list, exception_list)


@pytest.mark.parametrize(
    "min_df, max_df, exception_list",
    (
        pytest.param(
            pd.DataFrame(
                {"stack_1": [np.nan, 0, 0, 1], "stack_2": [np.nan, 0, 0, 0]},
            ),
            pd.DataFrame(
                {"stack_1": [np.nan, 0.5, 0.5, 1], "stack_2": [np.nan, 2, 0.5, 1]},
            ),
            [],
            id="zero exceptions",
        ),
        pytest.param(
            pd.DataFrame(
                {"stack_1": [np.nan, 2, 0, 1], "stack_2": [np.nan, 0, 1.5, 0]},
            ),
            pd.DataFrame(
                {"stack_1": [np.nan, 0.5, 0.5, 1], "stack_2": [np.nan, 2, 0.5, 1]},
            ),
            [
                NetworkValidatorException(
                    "Minimal fraction must be lower or equal to 1 for each year. "
                    "Detected values greater than 1 for years: [1, 2]"
                ),
                NetworkValidatorException(
                    "Minimal fraction sum must be lower than 1. Detected values "
                    "greater than 1 for years: [1, 2]"
                ),
            ],
            id="min fraction greater than 1",
        ),
        pytest.param(
            pd.DataFrame(
                {"stack_1": [np.nan, 0.5, 0, 1], "stack_2": [np.nan, 0.6, 0, 0.6]},
            ),
            pd.DataFrame(
                {"stack_1": [np.nan, 0.5, 0.5, 1], "stack_2": [np.nan, 2, 0.5, 1]},
            ),
            [
                NetworkValidatorException(
                    "Minimal fraction sum must be lower than 1. Detected values greater than 1 for years: [1, 3]"
                )
            ],
            id="min fraction sum greater than 1",
        ),
        pytest.param(
            pd.DataFrame(
                {"stack_1": [np.nan, 0, 0, 0], "stack_2": [np.nan, 0, 0, 0]},
            ),
            pd.DataFrame(
                {"stack_1": [np.nan, 0.5, 0.5, 0.4], "stack_2": [np.nan, 2, 0.5, 0.3]},
            ),
            [
                NetworkValidatorException(
                    "If maximal fraction is defined for every available stack in this aggregate consumer, "
                    "then sum of these fractions must not be lower than 1. Detected incorrect values for years: [3]"
                )
            ],
            id="max fraction sum lower than 1 for every stack",
        ),
        pytest.param(
            pd.DataFrame(
                {"stack_1": [np.nan, 0, 0, 1], "stack_2": [np.nan, 0, 0, 0]},
            ),
            pd.DataFrame(
                {"stack_1": [np.nan, 0.5, 0.5, 0.5]},
            ),
            [],
            id="max fraction not defined for every stack",
        ),
    ),
)
def test_validation_aggregated_fractions(
    aggregate_consumer: AggregatedConsumer,
    network: Network,
    min_df: pd.DataFrame,
    max_df: pd.DataFrame,
    exception_list: list[NetworkValidatorException],
) -> None:
    actual_exception_list: list[NetworkValidatorException] = []

    aggregate_consumer._validate_fraction_aggregated_values(
        min_df=min_df,
        max_df=max_df,
        exception_list=actual_exception_list,
    )

    assert_same_exception_list(actual_exception_list, exception_list)


@pytest.mark.parametrize(
    "min_fraction, max_fraction, max_fraction_increase, max_fraction_decrease, exception_list",
    (
        pytest.param(
            pd.Series([np.nan, 0, 0, 0]),
            pd.Series([np.nan, 1, 1, 0]),
            pd.Series([np.nan, 1, 0, 0]),
            pd.Series([np.nan, 0, 1, 0]),
            [],
            id="no exceptions",
        ),
        pytest.param(
            pd.Series([np.nan, 0.5, 0.2, 0.3]),
            pd.Series([np.nan, 1, 0.1, 0]),
            pd.Series([np.nan, 1, 1, 1]),
            pd.Series([np.nan, 1, 1, 1]),
            [
                NetworkValidatorException(
                    "Maximal fraction must be greater or equal than minimal fraction "
                    "value. Detected incorrect values for years: [2, 3]"
                )
            ],
            id="max fraction lower than min fraction",
        ),
        pytest.param(
            pd.Series([np.nan, 0, 0.9, 0]),
            pd.Series([np.nan, 0.2, 1, 1]),
            pd.Series([np.nan, 0.5, 0, 1]),
            pd.Series([np.nan, 1, 1, 1]),
            [
                NetworkValidatorException(
                    "In every year minimal fraction in the next year must be lower "
                    "than maximal fraction plus maximal fraction increase. Detected incorrect values for years: [1]"
                )
            ],
            id="max increase lower than needed",
        ),
        pytest.param(
            pd.Series([np.nan, 1, 0, 0]),
            pd.Series([np.nan, 1, 0.3, 0]),
            pd.Series([np.nan, 1, 1, 1]),
            pd.Series([np.nan, 0, 0.3, 0]),
            [
                NetworkValidatorException(
                    "In every year maximal fraction in the next year must be lower "
                    "than minimal fraction plus maximal fraction decrease. Detected incorrect values for years: [1]"
                )
            ],
            id="max decrease lower than needed",
        ),
    ),
)
def test_validation_stack_fractions(
    aggregate_consumer: AggregatedConsumer,
    network: Network,
    min_fraction: pd.Series | None,
    max_fraction: pd.Series | None,
    max_fraction_increase: pd.Series | None,
    max_fraction_decrease: pd.Series | None,
    exception_list: list[NetworkValidatorException],
) -> None:
    actual_exception_list: list[NetworkValidatorException] = []
    aggregate_consumer._validate_fraction_per_stack(
        min_fraction,
        max_fraction,
        max_fraction_increase,
        max_fraction_decrease,
        actual_exception_list,
    )

    assert_same_exception_list(actual_exception_list, exception_list)


@pytest.mark.parametrize(
    "stack_base_fraction, fraction, fraction_name, expected_exceptions",
    (
        (
            {"stack_1": 0, "stack_2": 1},
            {
                "stack_1": pd.Series([np.nan, 1, 0, 1]),
                "stack_2": pd.Series([np.nan, 1, 0, 1]),
            },
            "min_fraction",
            [],
        ),
        (
            {"stack_1": 0, "stack_2": 0},
            {
                "stack_1": pd.Series([np.nan, 1, 1, 0.3]),
                "stack_2": pd.Series([np.nan, 0, 0, 0.7]),
            },
            "max_fraction_increase",
            [
                NetworkValidatorException(
                    "For binary fraction setting, max_fraction_increase in stack stack_1 must contain "
                    "only values 0 or 1 for all years. Detected incorrect "
                    "values for years: [3]. Values: [0.3]"
                ),
                NetworkValidatorException(
                    "For binary fraction setting, max_fraction_increase in stack stack_2 must contain "
                    "only values 0 or 1 for all years. Detected incorrect "
                    "values for years: [3]. Values: [0.7]"
                ),
            ],
        ),
    ),
)
def test_binary_fraction(
    stack_base_fraction: dict[str, float],
    fraction: dict[str, pd.Series],
    fraction_name: str,
    expected_exceptions: list[NetworkValidatorException],
    network: Network,
    aggregate_consumer: AggregatedConsumer,
) -> None:
    network.constants = NetworkConstants(
        n_hours=network.constants.n_hours,
        n_years=network.constants.n_years,
        binary_fraction=True,
        relative_emission_limits={
            CO2_EMISSION: pd.Series([0.95, 0.85, 0.75], index=range(1, 4)),
            PM10_EMISSION: pd.Series([0.95, 0.85, 0.75], index=range(1, 4)),
        },
        base_total_emission={
            CO2_EMISSION: np.nan,
            PM10_EMISSION: np.nan,
        },
    )
    aggregate_consumer.stack_base_fraction = stack_base_fraction
    setattr(aggregate_consumer, fraction_name, fraction)
    actual_exception_list: list[NetworkValidatorException] = []

    aggregate_consumer._validate_fraction(fraction_name, network, actual_exception_list)

    assert_same_exception_list(actual_exception_list, expected_exceptions)


@pytest.mark.parametrize(
    "n_years, n_consumers, expected_exceptions",
    (
        (
            5,
            pd.Series([1, 2, 3, 4, 5]),
            [],
        ),
        (
            5,
            pd.Series([1, 2, 3]),
            [NetworkValidatorException("N_consumer series must have 5 values")],
        ),
        (
            5,
            pd.Series([0, 1, 2]),
            [
                NetworkValidatorException("N_consumer series must have 5 values"),
                NetworkValidatorException(
                    "For n_consumers series all values must be positive "
                    "and given for each year in simulation."
                ),
            ],
        ),
        (
            3,
            pd.Series([-1, 1, 2]),
            [
                NetworkValidatorException(
                    "For n_consumers series all values must be positive "
                    "and given for each year in simulation."
                ),
            ],
        ),
    ),
)
def test_n_consumer(
    n_years: int,
    n_consumers: pd.Series,
    expected_exceptions: list[NetworkValidatorException],
    aggregate_consumer: AggregatedConsumer,
) -> None:
    actual_exceptions: list[NetworkValidatorException] = []
    aggregate_consumer.n_consumers = n_consumers

    aggregate_consumer._validate_n_consumers(n_years, actual_exceptions)

    assert_same_exception_list(actual_exceptions, expected_exceptions)


@pytest.mark.parametrize(
    "aggregated_consumer, exception_list",
    (
        (
            AggregatedConsumer(
                name="aggregate_2",
                demand_profile="default",
                stack_base_fraction={"stack_3": 0.5, "stack_4": 0.5},
                yearly_energy_usage={
                    ELECTRICITY: pd.Series(data=range(5)),
                    HEATING: pd.Series(data=range(5)),
                },
                min_fraction={
                    "stack_3": pd.Series([np.nan] * default_network_constants.n_years),
                    "stack_4": pd.Series([np.nan] * default_network_constants.n_years),
                },
                max_fraction={
                    "stack_3": pd.Series([np.nan] * default_network_constants.n_years),
                    "stack_4": pd.Series([np.nan] * default_network_constants.n_years),
                },
                max_fraction_decrease={
                    "stack_3": pd.Series([np.nan] * default_network_constants.n_years),
                    "stack_4": pd.Series([np.nan] * default_network_constants.n_years),
                },
                max_fraction_increase={
                    "stack_3": pd.Series([np.nan] * default_network_constants.n_years),
                    "stack_4": pd.Series([np.nan] * default_network_constants.n_years),
                },
                n_consumers=pd.Series([1000] * default_network_constants.n_years),
                average_area=None,
            ),
            [
                NetworkValidatorException(
                    "Energy types of aggregated consumer aggregate_2 are different than "
                    "energy types defined in the connected stack stack_3. Difference: {'HEATING'}"
                ),
                NetworkValidatorException(
                    "Energy types of aggregated consumer demand profile aggregate_2 are different than energy"
                    " types defined in the connected stack stack_3. Difference: {'HEATING'}"
                ),
            ],
        ),
        (
            AggregatedConsumer(
                name="aggregate_3",
                demand_profile="default",
                stack_base_fraction={"stack_3": 1},
                yearly_energy_usage={
                    ELECTRICITY: pd.Series(data=range(5)),
                },
                min_fraction={
                    "stack_3": pd.Series([np.nan] * default_network_constants.n_years),
                },
                max_fraction={
                    "stack_3": pd.Series([np.nan] * default_network_constants.n_years),
                },
                max_fraction_decrease={
                    "stack_3": pd.Series([np.nan] * default_network_constants.n_years),
                },
                max_fraction_increase={
                    "stack_3": pd.Series([np.nan] * default_network_constants.n_years),
                },
                n_consumers=pd.Series([1000] * default_network_constants.n_years),
                average_area=None,
            ),
            [
                NetworkValidatorException(
                    "Energy types of aggregated consumer demand profile aggregate_3 are different than energy"
                    " types defined in the connected stack stack_3. Difference: {'HEATING'}"
                ),
            ],
        ),
    ),
)
def test_incomplete_stack_energy_types(
    aggregated_consumer: AggregatedConsumer,
    exception_list: list[NetworkValidatorException],
    network: Network,
) -> None:
    network.add_local_balancing_stack(
        LocalBalancingStack(
            name="stack_4",
            buses_out={ELECTRICITY: "bus_A", HEATING: "bus_C"},
            buses={ELECTRICITY: {"bus_A", "bus_B"}, HEATING: {"bus_C"}},
        )
    )
    network.add_local_balancing_stack(
        LocalBalancingStack(
            name="stack_3",
            buses_out={ELECTRICITY: "bus_A"},
            buses={ELECTRICITY: {"bus_A", "bus_B"}, HEATING: {"bus_C"}},
        )
    )

    with pytest.raises(NetworkValidatorExceptionGroup) as e_info:
        network.add_aggregated_consumer(aggregated_consumer)
    assert_same_exception_list(list(e_info.value.exceptions), exception_list)


@pytest.mark.parametrize(
    "aggregated_consumer, exception_list",
    (
        (
            AggregatedConsumer(
                name="aggregate_1",
                demand_profile="default",
                stack_base_fraction={"stack_3": 1.0},
                yearly_energy_usage={
                    ELECTRICITY: pd.Series(data=range(5)),
                    HEATING: pd.Series(data=range(5)),
                },
                min_fraction={
                    "stack_3": pd.Series([np.nan] * default_network_constants.n_years),
                },
                max_fraction={
                    "stack_3": pd.Series([np.nan] * default_network_constants.n_years),
                },
                max_fraction_decrease={
                    "stack_3": pd.Series([np.nan] * default_network_constants.n_years),
                },
                max_fraction_increase={
                    "stack_3": pd.Series([np.nan] * default_network_constants.n_years),
                },
                n_consumers=pd.Series([1000] * default_network_constants.n_years),
                average_area="string",  # noqa
            ),
            [
                NetworkValidatorException(
                    "Average area for AggregatedConsumer aggregate_1 must be given as float"
                )
            ],
        ),
    ),
)
def test_average_area(
    aggregated_consumer: AggregatedConsumer,
    exception_list: list[NetworkValidatorException],
    network: Network,
) -> None:
    network.add_local_balancing_stack(
        LocalBalancingStack(
            name="stack_3",
            buses_out={ELECTRICITY: "bus_A", HEATING: "bus_C"},
            buses={ELECTRICITY: {"bus_A", "bus_B"}, HEATING: {"bus_C"}},
        )
    )
    with pytest.raises(NetworkValidatorExceptionGroup) as e_info:
        aggregated_consumer.validate(network)
    assert_same_exception_list(list(e_info.value.exceptions), exception_list)
