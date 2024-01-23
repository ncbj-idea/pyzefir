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

import tempfile
from filecmp import dircmp
from pathlib import Path

import pytest

from pyzefir.utils.converters.xlsx_to_csv_converter import ExcelToCsvConverter
from tests.utils import get_resources


@pytest.fixture
def converted_xlsx_path() -> Path:
    input_files_path = get_resources("convert_input_structure")
    scenario_path = get_resources("convert_input_structure/scenarios/scenario_1.xlsx")
    with tempfile.TemporaryDirectory() as tmp_output:
        tmp_output_path = Path(tmp_output)
        converter = ExcelToCsvConverter(
            output_files_path=tmp_output_path,
            input_files_path=input_files_path,
            scenario_path=scenario_path,
        )

        converter.convert()

        yield tmp_output_path


def _check_diff_files(dircmp_: dircmp, diff_files: list[str]) -> None:
    for name in dircmp_.diff_files + dircmp_.right_only + dircmp_.left_only:
        diff_files.append(name)
    for sub_dcmp in dircmp_.subdirs.values():
        _check_diff_files(sub_dcmp, diff_files)


# This test is for now in unit tests, but as it may get slower consider putting this test in integration test directory
def test_check_csv_integrity(
    converted_xlsx_path: Path,
    csv_root_path: Path,
) -> None:
    dcmp = dircmp(converted_xlsx_path, csv_root_path)
    diff_files_list: list[str] = list()
    _check_diff_files(dcmp, diff_files_list)

    assert len(diff_files_list) == 0, f"Different files: {diff_files_list}"
