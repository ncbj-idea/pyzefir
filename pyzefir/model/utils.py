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

from dataclasses import dataclass
from types import UnionType

import numpy as np
import pandas as pd

from pyzefir.model.exceptions import NetworkValidatorException


@dataclass(frozen=True)
class NetworkConstants:
    n_years: int
    n_hours: int
    relative_emission_limits: dict[str, pd.Series]
    base_total_emission: dict[str, float | int]
    power_reserves: dict[str, dict[str, float | int]]
    min_generation_fraction: dict[str, dict[tuple[str, str], float]] | None = None
    max_generation_fraction: dict[str, dict[tuple[str, str], float]] | None = None
    binary_fraction: bool = False


def _check_if_series_has_numeric_values(series: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(series) and not pd.isnull(series).all()


def _validate_series(
    name: str,
    series: pd.Series,
    length: int,
    exception_list: list[NetworkValidatorException],
    is_numeric: bool = True,
    index_type: type | None = None,
    values_type: type | None = None,
    allow_null: bool = True,
) -> bool:
    is_validation_ok = True
    if is_numeric and _check_if_series_has_numeric_values(series):
        exception_list.append(
            NetworkValidatorException(f"{name} must have only numeric values")
        )
        is_validation_ok = False
    if len(series) != length:
        exception_list.append(
            NetworkValidatorException(f"{name} must have {length} values")
        )
        is_validation_ok = False
    if index_type and not np.issubdtype(series.index.dtype, index_type):
        exception_list.append(
            NetworkValidatorException(
                f"{name} index type is {series.index.dtype} but should be {index_type.__name__}"
            )
        )
        is_validation_ok = False
    if values_type and not np.issubdtype(series.dtype, values_type):
        exception_list.append(
            NetworkValidatorException(
                f"{name} type is {series.dtype} but should be {values_type.__name__}"
            )
        )
        is_validation_ok = False
    if not allow_null and pd.isnull(series).any():
        exception_list.append(
            NetworkValidatorException(f"{name} must not contain null values")
        )
        is_validation_ok = False
    return is_validation_ok


def validate_series(
    name: str,
    series: pd.Series,
    length: int,
    exception_list: list[NetworkValidatorException],
    is_numeric: bool = True,
    index_type: type | None = None,
    values_type: type | None = None,
    allow_null: bool = True,
) -> bool:
    """
    Validation procedure checking:
    - if series is a pandas series
    - if series has only numeric values
    - if series has correct length

    (Optional) Validation procedure also checking:
    - if series.index has provided type
    - if series has provided type
    - if series has null values
    """
    if isinstance(series, pd.Series):
        is_validation_ok = _validate_series(
            name=name,
            series=series,
            length=length,
            exception_list=exception_list,
            is_numeric=is_numeric,
            index_type=index_type,
            values_type=values_type,
            allow_null=allow_null,
        )
    else:
        exception_list.append(
            NetworkValidatorException(
                f"{name} must be a pandas Series, but {type(series).__name__} given"
            )
        )
        is_validation_ok = False
    return is_validation_ok


def validate_dict_type(
    dict_to_validate: dict,
    key_type: type | UnionType,
    value_type: type | UnionType,
    parameter_name: str,
    key_parameter_name: str,
    value_parameter_name: str,
    exception_list: list[NetworkValidatorException],
) -> bool:
    if not isinstance(dict_to_validate, dict):
        exception_list.append(
            NetworkValidatorException(
                f"{parameter_name.capitalize()} must be of dict type"
            )
        )
        return False

    is_validation_ok = True
    if not all(isinstance(key, key_type) for key in dict_to_validate.keys()):
        exception_list.append(
            NetworkValidatorException(
                f"{key_parameter_name.capitalize()} in {parameter_name} must be of "
                f"{key_type} type"
            )
        )
        is_validation_ok = False

    if not all(isinstance(value, value_type) for value in dict_to_validate.values()):
        exception_list.append(
            NetworkValidatorException(
                f"{value_parameter_name.capitalize()} in {parameter_name} must be "
                f"of {value_type} type"
            )
        )
        is_validation_ok = False

    return is_validation_ok


def check_interval(
    lower_bound: int | float,
    upper_bound: int | float,
    value: int | float,
    is_lower_bound_closed: bool = True,
    is_upper_bound_closed: bool = True,
) -> bool:
    """
    Checks if the given value falls within the specified interval defined by the lower and upper bounds.

    Args:
        lower_bound (int or float): The lower bound of the interval.
        upper_bound (int or float): The upper bound of the interval.
        value (int or float): The value to be checked against the interval.
        is_lower_bound_closed (bool, optional): Whether the lower bound is closed (inclusive). Default is True.
        is_upper_bound_closed (bool, optional): Whether the upper bound is closed (inclusive). Default is True.

    Returns:
        bool
    """

    return (
        (is_lower_bound_closed and lower_bound <= value)
        or (not is_lower_bound_closed and lower_bound < value)
    ) and (
        (is_upper_bound_closed and value <= upper_bound)
        or (not is_upper_bound_closed and value < upper_bound)
    )
