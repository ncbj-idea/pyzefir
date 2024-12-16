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

import numpy as np
import pandas as pd
import pytest

from pyzefir.model.network_elements import DSR
from pyzefir.parser.elements_parsers.dsr_parser import DSRParser

ParamValue = int | float | str


@pytest.mark.parametrize(
    ("parameters", "expected_result"),
    [
        pytest.param(
            {
                "name": "TEST_DSR",
                "compensation_factor": 0.7,
                "balancing_period_len": 24,
                "penalization_minus": 0.3,
                "penalization_plus": np.nan,
                "relative_shift_limit": np.nan,
                "abs_shift_limit": np.nan,
                "hourly_relative_shift_minus_limit": np.nan,
                "hourly_relative_shift_plus_limit": np.nan,
            },
            DSR(
                name="TEST_DSR",
                compensation_factor=0.7,
                balancing_period_len=24,
                penalization_minus=0.3,
                penalization_plus=0.0,
                relative_shift_limit=None,
                abs_shift_limit=None,
                hourly_relative_shift_minus_limit=1.0,
                hourly_relative_shift_plus_limit=1.0,
            ),
            id="no optional parameters",
        ),
        pytest.param(
            {
                "name": "TEST_DSR",
                "compensation_factor": 0.7,
                "balancing_period_len": 24,
                "penalization_minus": 0.3,
                "penalization_plus": 0.4,
                "relative_shift_limit": 0.3,
                "abs_shift_limit": 0.5,
                "hourly_relative_shift_minus_limit": 0.1,
                "hourly_relative_shift_plus_limit": 0.2,
            },
            DSR(
                name="TEST_DSR",
                compensation_factor=0.7,
                balancing_period_len=24,
                penalization_minus=0.3,
                penalization_plus=0.4,
                relative_shift_limit=0.3,
                abs_shift_limit=0.5,
                hourly_relative_shift_minus_limit=0.1,
                hourly_relative_shift_plus_limit=0.2,
            ),
            id="all optional parameters",
        ),
    ],
)
def test_create_single_dsr_obj(
    parameters: dict[str, ParamValue], expected_result: DSR
) -> None:
    result = DSRParser(pd.DataFrame([parameters])).create()[0]
    assert result == expected_result


def test_DSRParser_empty_dataframe() -> None:
    result = DSRParser(pd.DataFrame()).create()

    assert isinstance(result, tuple)
    assert not result
    assert len(result) == 0


@pytest.mark.parametrize(
    ("parameters", "error_message"),
    [
        pytest.param(
            {
                "compensation_factor": 0.7,
                "balancing_period_len": 24,
                "penalization_minus": 0.3,
                "penalization_plus": 0.4,
                "relative_shift_limit": np.nan,
                "abs_shift_limit": np.nan,
                "hourly_relative_shift_minus_limit": np.nan,
                "hourly_relative_shift_plus_limit": np.nan,
            },
            "name",
            id="no name parameters",
        ),
        pytest.param(
            {
                "name": "TEST_DSR",
                "balancing_period_len": 24,
                "penalization_minus": 0.3,
                "penalization_plus": 0.4,
                "relative_shift_limit": np.nan,
                "abs_shift_limit": np.nan,
                "hourly_relative_shift_minus_limit": np.nan,
                "hourly_relative_shift_plus_limit": np.nan,
            },
            "compensation_factor",
            id="no compensation_factor parameters",
        ),
        pytest.param(
            {
                "name": "TEST_DSR",
                "balancing_period_len": 24,
                "compensation_factor": 0.7,
                "penalization_minus": 0.3,
                "penalization_plus": 0.5,
                "relative_shift_limit": np.nan,
                "abs_shift_limit": np.nan,
                "hourly_relative_shift_minus_limit": np.nan,
            },
            "hourly_relative_shift_plus_limit",
            id="no hourly_relative_shift_plus_limit parameters",
        ),
        pytest.param(
            {
                "name": "TEST_DSR",
                "balancing_period_len": 24,
                "compensation_factor": 0.7,
                "penalization_minus": 0.3,
                "relative_shift_limit": np.nan,
                "abs_shift_limit": np.nan,
                "hourly_relative_shift_minus_limit": np.nan,
            },
            "penalization_plus",
            id="no penalization_plus parameters",
        ),
    ],
)
def test_DSRParser_error_input(
    parameters: dict[str, ParamValue], error_message: str
) -> None:
    with pytest.raises(KeyError, match=error_message):
        DSRParser(pd.DataFrame([parameters])).create()
