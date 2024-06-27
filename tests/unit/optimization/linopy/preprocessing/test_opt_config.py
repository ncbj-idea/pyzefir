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
from typing import Any, Type

import numpy as np
import pytest
from numpy import arange, array, in1d, ndarray, ones, unique

from pyzefir.optimization.opt_config import OptConfig, OptConfigError


@pytest.mark.parametrize(
    (
        "kwargs",
        "error",
        "error_msg",
    ),
    [
        (
            {
                "hours": 100,
                "years": array([[1]]),
                "hour_sample": None,
                "year_sample": None,
                "discount_rate": arange(5),
            },
            OptConfigError,
            "discount_rate, hours and years must be 1D arrays",
        ),
        (
            {
                "hours": array([[1]]),
                "years": 10,
                "hour_sample": None,
                "year_sample": None,
                "discount_rate": arange(5),
            },
            OptConfigError,
            "discount_rate, hours and years must be 1D arrays",
        ),
        (
            {
                "hours": 20,
                "years": 10,
                "hour_sample": 20,
                "year_sample": 10,
                "discount_rate": array([[1]]),
            },
            OptConfigError,
            "discount_rate, hours and years must be 1D arrays",
        ),
        (
            {
                "hours": 20,
                "years": 10,
                "hour_sample": 21,
                "year_sample": 10,
                "discount_rate": arange(10),
            },
            ValueError,
            "Cannot take a larger sample than population when 'replace=False'",
        ),
        (
            {
                "hours": 10,
                "years": 10,
                "hour_sample": 10,
                "year_sample": 10,
                "discount_rate": arange(4),
            },
            OptConfigError,
            "discount_rate shape is different than years shape",
        ),
        (
            {
                "hours": 10,
                "years": 10,
                "hour_sample": 10,
                "year_sample": array([0, 2, 3, 5]),
                "discount_rate": arange(10),
            },
            OptConfigError,
            "year sample must be consecutive starting from 0",
        ),
        (
            {
                "hours": 10,
                "years": 10,
                "hour_sample": 10,
                "year_sample": array([3, 4, 5]),
                "discount_rate": arange(10),
            },
            OptConfigError,
            "year sample must be consecutive starting from 0",
        ),
        (
            {
                "hours": 10,
                "years": 10,
                "hour_sample": 10,
                "year_sample": 15,
                "discount_rate": arange(10),
            },
            OptConfigError,
            "year sample 15 must be less than or equal to year shape 10",
        ),
        (
            {
                "hours": 10,
                "years": 15,
                "hour_sample": 10,
                "year_sample": 15,
                "discount_rate": arange(15),
                "generator_capacity_cost": "not-correct",
            },
            OptConfigError,
            "generator capacity cost should be 'brutto' or 'netto'",
        ),
    ],
)
def test_errors(
    kwargs: dict[str, Any],
    error: Type[Exception],
    error_msg: str,
) -> None:
    with pytest.raises(error) as err:
        OptConfig(**kwargs)
    if isinstance(err.value, ExceptionGroup):
        assert error_msg in [str(e) for e in err.value.exceptions]
    else:
        assert error_msg == str(err.value)


@pytest.mark.parametrize(
    ("idx", "sample", "expected_result"),
    [
        (arange(100), array([2, 3, 4, 5]), array([2, 3, 4, 5])),
        (array([1, 2, 5, 7, 8]), array([2, 0, 4]), array([2, 0, 4])),
        (arange(100), None, arange(100)),
    ],
)
def test_get_sample(
    idx: ndarray, sample: int | ndarray | None, expected_result: ndarray
) -> None:
    assert np.all(OptConfig.get_sample(idx, sample) == expected_result)


@pytest.mark.parametrize(
    ("sample", "expected_exception_msg"),
    (
        (5, None),
        (np.array([5, 3, 2]), None),
        (None, None),
        (
            "5",
            "sample must be <class 'int'>, <class 'numpy.ndarray'> or None, but is of type <class 'str'>",
        ),
        (
            [5, 3, 2],
            "sample must be <class 'int'>, <class 'numpy.ndarray'> or None, but is of type <class 'list'>",
        ),
    ),
)
def test_get_sample_incorrect_type(
    sample: Any, expected_exception_msg: str | None
) -> None:
    idx = np.arange(10)

    if expected_exception_msg is not None:
        with pytest.raises(OptConfigError) as exc:
            OptConfig.get_sample(idx, sample)
        assert str(exc.value) == expected_exception_msg
    else:
        result = OptConfig.get_sample(idx, sample)
        assert result is not None


@pytest.mark.parametrize(
    ("idx", "sample_size"),
    [
        (arange(100), 10),
        (arange(10), 10),
        (array([2, 3, 10, 2]), 4),
        (array([1]), 1),
        (arange(10), 0),
    ],
)
def test_get_sample_random(idx: ndarray, sample_size: int) -> None:
    result = OptConfig.get_sample(idx, sample_size)

    assert np.all(in1d(result, arange(idx.shape[0])))
    assert result.shape == (sample_size,)
    assert unique(result).shape == result.shape


@pytest.mark.parametrize(
    (
        "hours",
        "years",
        "year_sample",
        "hour_sample",
        "discount_rate",
        "expected_hours",
        "expected_years",
        "expected_hours_ratio",
    ),
    [
        (8760, 20, arange(50), None, ones(20) * 0.5, arange(8760), arange(20), 1.0),
        (8760, 20, None, None, ones(20) * 0.02, arange(8760), arange(20), 1.0),
        (8760, 20, None, arange(450), ones(20) * 0.02, arange(8760), arange(20), 19.4),
    ],
)
def test_init(
    hours: int | ndarray,
    years: int | ndarray,
    year_sample: int | ndarray | None,
    hour_sample: int | ndarray | None,
    discount_rate: ndarray,
    expected_hours: ndarray,
    expected_years: ndarray,
    expected_hours_ratio: float,
) -> None:
    opt_config = OptConfig(hours, years, discount_rate, hour_sample, year_sample)

    assert np.all(opt_config.hours == expected_hours)
    assert np.all(opt_config.years == expected_years)
    assert np.all(
        opt_config.hour_sample == opt_config.get_sample(opt_config.hours, hour_sample)
    )
    assert np.all(
        opt_config.year_sample == opt_config.get_sample(opt_config.years, year_sample)
    )
    assert np.all(opt_config.discount_rate == discount_rate)
    assert opt_config.hourly_scale == pytest.approx(expected_hours_ratio, 0.1)
