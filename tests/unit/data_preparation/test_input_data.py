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

from dataclasses import fields
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pyzefir.structure_creator.data_loader.input_data import (
    InputData,
    InputStructureData,
    ScenarioData,
)


@pytest.fixture
def mock_read_excel(*args: Any, **kwargs: Any) -> dict[str, pd.DataFrame]:
    return {"MockExcelData": pd.DataFrame(data={"Idx": [1, 2], "Col": ["A", "B"]})}


@pytest.fixture
def mock_load_fractions(
    *args: Any, **kwargs: Any
) -> dict[str, dict[str, pd.DataFrame]]:
    return {
        "MockFraction": {
            "MockCategory": pd.DataFrame(data={"Idx": [3, 4], "Col": ["D", "B"]})
        }
    }


def test_load_scenario_data(
    mock_read_excel: dict[str, pd.DataFrame],
    mock_load_fractions: dict[str, dict[str, pd.DataFrame]],
) -> None:
    with patch(
        "pyzefir.structure_creator.data_loader.input_data.pd.read_excel",
        return_value=mock_read_excel,
    ), patch.object(ScenarioData, "_load_fractions", return_value=mock_load_fractions):
        scenario_data = ScenarioData.load_scenario_data(Path("/mocked/path"))

        assert isinstance(scenario_data, ScenarioData)

        for field in fields(scenario_data):
            if field.name != "fractions":
                actual_value = getattr(scenario_data, field.name)
                check_excel_file(
                    expected_excel=mock_read_excel,
                    actual_value=actual_value,
                )
        for key, value in scenario_data.fractions.items():
            assert key in mock_load_fractions
            for inner_key, df in value.items():
                assert inner_key in mock_load_fractions[key]
                assert_frame_equal(df, mock_load_fractions[key][inner_key])


def test_load_structure_data(
    mock_read_excel: dict[str, pd.DataFrame],
) -> None:
    with patch(
        "pyzefir.structure_creator.data_loader.input_data.pd.read_excel",
        return_value=mock_read_excel,
    ), patch.object(
        InputStructureData,
        "_load_lbs_files",
        return_value=mock_read_excel,
    ):
        structure_data = InputStructureData.load_structure_data(
            Path("/mocked/path"), 24, 5
        )

        for field in fields(structure_data):
            if field.name in ["n_hours", "n_years"]:
                continue
            actual_value = getattr(structure_data, field.name)
            check_excel_file(
                expected_excel=mock_read_excel,
                actual_value=actual_value,
            )

        assert structure_data.n_hours == 24
        assert structure_data.n_years == 5


def test_load_input_data() -> None:
    with patch.object(
        ScenarioData, "load_scenario_data", return_value="ScenarioData"
    ), patch.object(
        InputStructureData, "load_structure_data", return_value="StructureData"
    ):
        input_data = InputData.load_input_data(
            Path("/mocked/path"), "scenario_name", 24, 5
        )

        assert isinstance(input_data, InputData)
        assert input_data.scenario_data == "ScenarioData"
        assert input_data.structure_data == "StructureData"


def check_excel_file(
    expected_excel: dict[str, pd.DataFrame], actual_value: dict[str, pd.DataFrame]
) -> None:
    assert isinstance(actual_value, dict)
    assert actual_value.keys() == expected_excel.keys()
    assert isinstance(actual_value["MockExcelData"], pd.DataFrame)
    assert_frame_equal(
        left=actual_value["MockExcelData"], right=expected_excel["MockExcelData"]
    )
