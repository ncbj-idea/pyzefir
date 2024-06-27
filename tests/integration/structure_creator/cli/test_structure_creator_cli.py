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

import shutil
import tempfile
from pathlib import Path
from typing import Any, Generator

import pandas as pd
import pytest
from click.testing import CliRunner

from pyzefir.structure_creator.cli.cli_wrapper import (
    create_structure,
    run_structure_creator_cli,
)
from tests.utils import RESOURCES_PATH


@pytest.fixture
def output_path() -> Generator[str, Any, None]:
    """Temporary output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield str(temp_dir)


@pytest.fixture
def temporary_directory() -> Generator[Path, Any, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        resource_path = Path(RESOURCES_PATH) / "structure_creator_resources"

        for item in resource_path.iterdir():
            if item.is_dir():
                shutil.copytree(item, temp_path / item.name)
            else:
                shutil.copy2(item, temp_path / item.name)

        yield temp_path


def test_main(output_path: str) -> None:
    runner = CliRunner()
    runner.invoke(
        run_structure_creator_cli,
        args=[
            "--input_path",
            str(Path(RESOURCES_PATH) / "structure_creator_resources"),
            "--output_path",
            output_path,
            "--scenario_name",
            "base_variant",
            "--n_hours",
            "100",
            "--n_years",
            "2",
        ],
        catch_exceptions=False,
    )


def test_main_without_yearly_emission_reduction(
    output_path: str, temporary_directory: Path
) -> None:
    yer_file_path = (
        temporary_directory
        / "scenarios"
        / "base_variant"
        / "yearly_emission_reduction.xlsx"
    )
    if yer_file_path.exists():
        yer_file_path.unlink()

    create_structure(
        input_path=str(temporary_directory),
        output_path=output_path,
        scenario_name="base_variant",
        n_hours=100,
        n_years=2,
    )
    df_dict = pd.read_excel(
        Path(output_path) / "scenarios" / "base_variant.xlsx", sheet_name=None
    )
    assert "Yearly Emission Reduction" not in df_dict


def test_main_without_capacity_bounds(
    output_path: str, temporary_directory: Path
) -> None:
    folder_path = temporary_directory / "lbs"
    for file_path in folder_path.glob("*.xlsx"):
        df_dict = pd.read_excel(file_path, sheet_name=None)
        if "CAPACITY_BOUNDS" in df_dict:
            df_dict.pop("CAPACITY_BOUNDS")
            with pd.ExcelWriter(file_path) as writer:
                for sheet, df in df_dict.items():
                    df.to_excel(writer, sheet_name=sheet, index=False)

    create_structure(
        input_path=str(temporary_directory),
        output_path=output_path,
        scenario_name="base_variant",
        n_hours=100,
        n_years=2,
    )
    df_dict = pd.read_excel(
        Path(output_path) / "scenarios" / "base_variant.xlsx", sheet_name=None
    )
    assert "Capacity Bounds" not in df_dict
