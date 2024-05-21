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
from pathlib import Path
from typing import Any

from click.testing import CliRunner
from linopy import Model, solvers
from pytest_mock import MockFixture

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
        "generator_type_efficiency",
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
        csv_dump_path.exists()
        and csv_dump_path.is_dir()
        and csv_dump_dir_expected_content == csv_dump_dir_content
    )
    assert (output_path / "cli.log").exists() and (output_path / "cli.log").is_file()
    assert (output_path / "csv" / "Objective_func_value.csv").is_file()
    assert (
        set(x.name for x in (output_path / "csv").iterdir() if x.is_dir())
        == results_expected_dirs
    )


def test_simple_run_no_storages(
    config_ini_path: Path,
    config_parser_no_storages: configparser.ConfigParser,
    output_path: Path,
    csv_dump_path: Path,
) -> None:
    """In this test also the generator efficiency is not presented with an additional data series file"""
    set_up_config_ini(config_ini_path, config_parser_no_storages)
    runner = CliRunner()
    result = runner.invoke(
        cli_run, ["--config", str(config_ini_path)], catch_exceptions=False
    )

    csv_dump_dir_expected_content = {
        "capacity_factors",
        "demand_chunks",
        "demand_types",
        "fuels",
        "conversion_rate",
        "generator_types",
        "initial_state",
        "scenarios",
        "structure",
    }
    results_expected_dirs = {
        "lines_results",
        "generators_results",
        "fractions_results",
        "bus_results",
    }
    csv_dump_dir_content = set(x.name for x in csv_dump_path.iterdir())

    assert result.exit_code == 0
    assert (
        csv_dump_path.exists()
        and csv_dump_path.is_dir()
        and csv_dump_dir_expected_content == csv_dump_dir_content
    )
    assert (output_path / "cli.log").exists() and (output_path / "cli.log").is_file()
    assert (output_path / "csv" / "Objective_func_value.csv").is_file()
    assert (
        set(x.name for x in (output_path / "csv").iterdir() if x.is_dir())
        == results_expected_dirs
    )


def test_simple_run_with_creator(
    config_ini_path: Path,
    config_parser_with_creator: configparser.ConfigParser,
) -> None:
    set_up_config_ini(config_ini_path, config_parser_with_creator)
    runner = CliRunner()
    runner.invoke(cli_run, ["--config", str(config_ini_path)], catch_exceptions=False)


def test_simple_run_and_xlsx_dump(
    config_ini_path: Path,
    config_parser_with_xlsx: configparser.ConfigParser,
    output_path: Path,
    csv_dump_path: Path,
) -> None:
    set_up_config_ini(config_ini_path, config_parser_with_xlsx)
    runner = CliRunner()
    result = runner.invoke(
        cli_run, ["--config", str(config_ini_path)], catch_exceptions=False
    )

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
        "generator_type_efficiency",
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


def test_custom_solver_args(
    config_ini_path: Path,
    config_parser: configparser.ConfigParser,
    output_path: Path,
    csv_dump_path: Path,
    mocker: MockFixture,
) -> None:
    config = {s: dict(config_parser.items(s)) for s in config_parser.sections()}
    for solver_name in solvers.available_solvers:
        config[solver_name] = {}
        config[solver_name]["MIPGap"] = "-3"
        config[solver_name]["LogFile"] = "test.log"
        config[solver_name]["TestFLOAT"] = "0.243"

    new_config_parser = configparser.ConfigParser()
    new_config_parser.optionxform = str  # type: ignore
    new_config_parser.read_dict(config)

    set_up_config_ini(config_ini_path, new_config_parser)
    runner = CliRunner()

    results_kwargs = {}

    def mock_solve(_self: Model, *_args: list[Any], **kwargs: dict[str, Any]) -> None:
        nonlocal results_kwargs
        results_kwargs = kwargs
        raise Exception("Mocked exception to stop the optimization")

    fake_model = type("FakeModel", (Model,), {"solve": mock_solve})()

    mocker.patch(
        "pyzefir.optimization.linopy.model.LinopyOptimizationModel.model", fake_model
    )

    runner.invoke(cli_run, ["--config", str(config_ini_path)], catch_exceptions=True)

    for key, val in {
        "MIPGap": -3,
        "LogFile": "test.log",
        "TestFLOAT": 0.243,
    }.items():
        assert results_kwargs[key] == val
