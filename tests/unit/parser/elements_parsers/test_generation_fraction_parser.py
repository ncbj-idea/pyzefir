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

from pyzefir.model.network_elements.generation_fraction import GenerationFraction
from pyzefir.parser.elements_parsers.generation_fraction_parser import (
    GenerationFractionParser,
    GenerationFractionParserException,
)
from pyzefir.utils.path_manager import DataCategories, DataSubCategories


@pytest.fixture
def generation_fraction_df(csv_root_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_root_path
        / f"{DataCategories.SCENARIO}/scenario_1"
        / f"{DataSubCategories.GENERATION_FRACTION}.csv"
    )


@pytest.fixture
def generation_fraction_parser(
    generation_fraction_df: pd.DataFrame,
) -> GenerationFractionParser:
    return GenerationFractionParser(
        generation_fraction_df=generation_fraction_df,
        n_years=5,
    )


def test_generation_fraction_parser_init(
    generation_fraction_parser: GenerationFractionParser,
) -> None:
    assert isinstance(generation_fraction_parser.generation_fraction_df, pd.DataFrame)
    assert isinstance(generation_fraction_parser.n_years, int)


@pytest.mark.parametrize(
    "name, unique_values",
    [
        pytest.param(
            "test1", (np.array([1]), np.array([2])), id="single_unique_values"
        ),
        pytest.param(
            "test2", (np.array([1, 1]), np.array([2])), id="duplicate_in_first_array"
        ),
        pytest.param(
            "test3", (np.array([1]), np.array([2, 2])), id="duplicate_in_second_array"
        ),
        pytest.param(
            "test4",
            (np.array([1]), np.array([2]), np.array([3])),
            id="multiple_single_unique_values",
        ),
    ],
)
def test_validate_unique_values(name: str, unique_values: tuple[np.ndarray]) -> None:
    if any(len(unique_value) > 1 for unique_value in unique_values):
        with pytest.raises(GenerationFractionParserException):
            GenerationFractionParser._validate_unique_values(name, unique_values)
    else:
        GenerationFractionParser._validate_unique_values(name, unique_values)


@pytest.mark.parametrize(
    "df, column_name, n_years, expected_series",
    [
        pytest.param(
            pd.DataFrame({"year": [0, 1, 2], "value": [0.1, 0.2, 0.3]}),
            "value",
            5,
            pd.Series(
                [0.1, 0.2, 0.3, np.nan, np.nan],
                index=pd.Index(range(5), name="year"),
                name="value",
            ),
            id="basic_case_with_nans",
        ),
        pytest.param(
            pd.DataFrame({"year": [0, 1, 2], "value": [0.1, 0.2, 0.3]}),
            "value",
            3,
            pd.Series(
                [0.1, 0.2, 0.3], index=pd.Index(range(3), name="year"), name="value"
            ),
            id="exact_years_no_nans",
        ),
        pytest.param(
            pd.DataFrame({"year": [1, 2, 4], "value": [0.2, 0.3, 0.5]}),
            "value",
            6,
            pd.Series(
                [np.nan, 0.2, 0.3, np.nan, 0.5, np.nan],
                index=pd.Index(range(6), name="year"),
                name="value",
            ),
            id="missing_years",
        ),
        pytest.param(
            pd.DataFrame({"year": [5, 6], "value": [0.2, 0.2]}),
            "value",
            4,
            pd.Series(
                [np.nan, np.nan, np.nan, np.nan],
                index=pd.Index(range(4), name="year"),
                name="value",
            ),
            id="dataframe_out_of_range",
        ),
    ],
)
def test_create_fraction_series(
    df: pd.DataFrame, column_name: str, n_years: int, expected_series: pd.Series
) -> None:
    result_series = GenerationFractionParser._create_fraction_series(
        df, column_name, n_years
    )
    pd.testing.assert_series_equal(result_series, expected_series)


def test_generation_fraction_parser_create(
    generation_fraction_parser: GenerationFractionParser,
) -> None:
    generation_fractions = generation_fraction_parser.create()
    assert isinstance(generation_fractions, tuple)
    assert all(isinstance(gf, GenerationFraction) for gf in generation_fractions)
