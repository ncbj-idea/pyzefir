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
from pathlib import Path

from click.testing import CliRunner

from pyzefir.cli.runner import cli_run


def set_up_config_ini(path: Path, config_parser: configparser.ConfigParser) -> None:
    with open(path, mode="w") as file_handler:
        config_parser.write(file_handler)


def test_simple_run(
    config_ini_path: Path,
    config_parser: configparser.ConfigParser,
    output_path: Path,
    csv_dump_path: Path,
) -> None:
    set_up_config_ini(config_ini_path, config_parser)
    runner = CliRunner()
    result = runner.invoke(
        cli_run, ["--config", str(config_ini_path)], catch_exceptions=False
    )

    sol_file_path, log_file_path = output_path / "file.sol", output_path / "file.log"
    csv_dump_dir_expected_content = {
        "capacity_factors",
        "demand_chunks",
        "demand_types",
        "fuels",
        "conversion_rate",
        "generator_types",
        "initial_state",
        "scenarios",
        "storage_types",
        "structure",
    }
    results_expected_dirs = {
        "lines_results",
        "generators_results",
        "storages_results",
        "fractions_results",
        "bus_results",
    }
    csv_dump_dir_content = set(x.name for x in csv_dump_path.iterdir())

    assert result.exit_code == 0
    assert (
        sol_file_path.exists()
        and sol_file_path.is_file()
        and os.stat(sol_file_path).st_size > 0
    )
    assert (
        log_file_path.exists()
        and log_file_path.is_file()
        and os.stat(log_file_path).st_size > 0
    )
    assert (
        csv_dump_path.exists()
        and csv_dump_path.is_dir()
        and csv_dump_dir_expected_content == csv_dump_dir_content
    )
    assert (output_path / "cli.log").exists() and (output_path / "cli.log").is_file()
    assert (output_path / "csv" / "Objective_func_value.csv").is_file() and (
        output_path / "xlsx" / "Objective_func_value.xlsx"
    ).is_file()
    assert (
        set(x.name for x in (output_path / "csv").iterdir() if x.is_dir())
        == results_expected_dirs
    )
    assert (
        set(x.name for x in (output_path / "xlsx").iterdir() if x.is_dir())
        == results_expected_dirs
    )


def test_simple_run_with_creator(
    config_ini_path: Path,
    config_parser_with_creator: configparser.ConfigParser,
) -> None:
    set_up_config_ini(config_ini_path, config_parser_with_creator)
    runner = CliRunner()
    runner.invoke(cli_run, ["--config", str(config_ini_path)], catch_exceptions=False)
