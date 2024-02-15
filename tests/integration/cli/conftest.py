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

import configparser
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest
from linopy import solvers

root_input_path = Path(__file__).parent.parent.parent / "resources" / "integration_test"
input_path = root_input_path / "simple-data-poland"

parameters_path = root_input_path / "parameters"


@pytest.fixture
def tmp_dir_with_files() -> Path:
    input_path_with_creator = root_input_path / "simple-data-poland-creator"
    tmp_dir = Path(tempfile.mkdtemp())
    for element in os.listdir(str(input_path_with_creator)):
        source_path = os.path.join(input_path_with_creator, element)
        destiny_path = os.path.join(tmp_dir, element)
        if os.path.isdir(source_path):
            shutil.copytree(source_path, destiny_path)
        else:
            shutil.copy2(source_path, destiny_path)
    yield tmp_dir


@pytest.fixture(params=solvers.available_solvers)
def solver(request: Any) -> str:
    return request.param


@pytest.fixture
def config_parser(
    output_path: Path, csv_dump_path: Path, solver: str
) -> configparser.ConfigParser:
    """Simple configuration file for pipeline test run."""
    config = configparser.ConfigParser()
    config.read_dict(
        {
            "input": {
                "input_path": str(input_path),
                "input_format": "xlsx",
                "scenario": "scenario_1",
            },
            "output": {
                "output_path": str(output_path),
                "sol_dump_path": str(output_path / "file.sol"),
                "opt_logs_path": str(output_path / "file.log"),
                "csv_dump_path": str(csv_dump_path),
            },
            "parameters": {
                "hour_sample": str(parameters_path / "hour_sample.csv"),
                "year_sample": str(parameters_path / "year_sample.csv"),
                "discount_rate": str(parameters_path / "discount_rate_not_creator.csv"),
            },
            "optimization": {
                "binary_fraction": False,
                "money_scale": 100.0,
                "ens": False,
                "use_hourly_scale": True,
                "solver": solver,
            },
        }
    )
    return config


@pytest.fixture
def config_parser_with_creator(
    output_path: Path,
    csv_dump_path: Path,
    tmp_dir_with_files: Path,
) -> configparser.ConfigParser:
    """Simple configuration file for pipeline test run."""
    config = configparser.ConfigParser()
    config.read_dict(
        {
            "input": {
                "input_path": str(tmp_dir_with_files),
                "input_format": "xlsx",
                "scenario": "scenario_1",
            },
            "output": {
                "output_path": str(output_path),
                "sol_dump_path": str(output_path / "file.sol"),
                "opt_logs_path": str(output_path / "file.log"),
                "csv_dump_path": str(csv_dump_path),
            },
            "parameters": {
                "hour_sample": str(parameters_path / "hour_sample.csv"),
                "year_sample": str(parameters_path / "year_sample.csv"),
                "discount_rate": str(
                    parameters_path / "discount_rate_with_creator.csv"
                ),
            },
            "optimization": {
                "binary_fraction": False,
                "money_scale": 1000.0,
                "ens": True,
                "use_hourly_scale": True,
            },
            "create": {
                "n_hours": 8760,
                "n_years": 20,
                "input_path": str(tmp_dir_with_files / "structure_creator_resources"),
            },
        }
    )
    return config


@pytest.fixture
def config_ini_path(config_parser: configparser.ConfigParser) -> Path:
    """Create *.ini file."""
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".ini", delete=False
    ) as temp_file:
        yield Path(temp_file.name)


@pytest.fixture
def output_path() -> Path:
    """Temporary output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def csv_dump_path() -> Path:
    """Temporary directory for storing converted *.csv files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)
