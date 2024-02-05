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

from pyzefir.model.exceptions import NetworkValidatorException
from pyzefir.model.utils import check_interval, validate_series
from tests.unit.model.test_network_elements.helpers import assert_same_exception_list


@pytest.mark.parametrize(
    "kwargs, exception_list",
    [
        pytest.param(
            {"name": "test_series", "series": pd.Series([1, 2, 3]), "length": 3},
            [],
            id="correct_series",
        ),
        pytest.param(
            {
                "name": "test_series",
                "series": pd.Series([1, 2, "3"]),
                "length": 3,
            },
            [NetworkValidatorException("test_series must have only numeric values")],
            id="series_with_non_numeric_values",
        ),
        pytest.param(
            {
                "name": "test_series",
                "series": pd.Series([1, 2, "3"]),
                "length": 3,
                "is_numeric": False,
            },
            [],
            id="series_with_non_numeric_values_AND_is_numeric_is_false",
        ),
        pytest.param(
            {"name": "test_series", "series": pd.Series([1, 2, 3]), "length": 2},
            [NetworkValidatorException("test_series must have 2 values")],
            id="series_with_incorrect_length",
        ),
        pytest.param(
            {"name": "test_series", "series": pd.Series([1, 2, "3"]), "length": 2},
            [
                NetworkValidatorException("test_series must have only numeric values"),
                NetworkValidatorException("test_series must have 2 values"),
            ],
            id="series_with_incorrect_length_and_non_numeric_values",
        ),
        pytest.param(
            {"name": "test_series", "series": [1, 2, 3], "length": 3},
            [
                NetworkValidatorException(
                    "test_series must be a pandas Series, but list given"
                )
            ],
            id="series_is_not_a_pandas_series",
        ),
        pytest.param(
            {"name": "test_series", "series": None, "length": 3},
            [
                NetworkValidatorException(
                    "test_series must be a pandas Series, but NoneType given"
                )
            ],
            id="series_is_none",
        ),
        pytest.param(
            {
                "name": "test_series",
                "series": pd.Series([1, "3", None]),
                "length": 3,
                "is_numeric": False,
                "allow_null": False,
            },
            [NetworkValidatorException("test_series must not contain null values")],
            id="series_with_none_values",
        ),
        pytest.param(
            {
                "name": "test_series",
                "series": pd.Series([1, 3.2, np.nan]),
                "length": 3,
                "allow_null": False,
            },
            [NetworkValidatorException("test_series must not contain null values")],
            id="series_with_null_values",
        ),
    ],
)
def test_validate_series(
    kwargs: dict, exception_list: list[NetworkValidatorException]
) -> None:
    """
    Test if validate_series works correctly
    """
    actual_exception_list: list[NetworkValidatorException] = []

    validate_series(**kwargs, exception_list=actual_exception_list)

    assert_same_exception_list(actual_exception_list, exception_list)


@pytest.mark.parametrize(
    "value, is_lower_closed, is_upper_closed",
    [
        pytest.param(
            -2,
            True,
            True,
            id="value_below_lower_bound_interval_closed",
        ),
        pytest.param(
            0,
            False,
            True,
            id="value_below_lower_bound_interval_open",
        ),
        pytest.param(
            150,
            True,
            True,
            id="value_below_lower_bound_interval_closed",
        ),
        pytest.param(
            3,
            True,
            False,
            id="value_below_lower_bound_interval_closed",
        ),
    ],
)
def test_check_interval(
    value: int | float,
    is_lower_closed: bool,
    is_upper_closed: bool,
) -> None:
    lower_bound = 0
    upper_bound = 3

    assert (
        check_interval(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            value=value,
            is_lower_bound_closed=is_lower_closed,
            is_upper_bound_closed=is_upper_closed,
        )
        is False
    )
