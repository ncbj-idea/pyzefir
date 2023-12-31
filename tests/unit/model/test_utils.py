import numpy as np
import pandas as pd
import pytest

from pyzefir.model.exceptions import NetworkValidatorException
from pyzefir.model.utils import validate_series
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
