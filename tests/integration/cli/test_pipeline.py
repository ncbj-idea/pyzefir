import configparser
from pathlib import Path
from typing import Any

import pytest
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
    assert (output_path / "model.lp").exists() and (output_path / "model.lp").is_file()


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


def test_simple_run_with_sol_file(
    config_ini_path: Path,
    config_parser: configparser.ConfigParser,
    output_path: Path,
    csv_dump_path: Path,
    sol_dump_path: Path,
) -> None:
    config = config_parser
    config["output"]["sol_dump_path"] = str(sol_dump_path / "solution.sol")
    set_up_config_ini(config_ini_path, config_parser)
    runner = CliRunner()
    runner.invoke(cli_run, ["--config", str(config_ini_path)], catch_exceptions=False)
    assert Path(config["output"]["sol_dump_path"]).exists()


def test_simple_run_gurobi_parameters_export(
    config_ini_path: Path,
    config_parser: configparser.ConfigParser,
    output_path: Path,
) -> None:
    gurobi_parameter_path = output_path / "gurobi_parameters.csv"
    config_parser["output"]["gurobi_parameters_path"] = str(gurobi_parameter_path)
    set_up_config_ini(config_ini_path, config_parser)
    runner = CliRunner()
    if config_parser["optimization"]["solver"] == "gurobi":
        runner.invoke(
            cli_run, ["--config", str(config_ini_path)], catch_exceptions=False
        )
        assert gurobi_parameter_path.exists()
        assert gurobi_parameter_path.is_file()
    else:
        error_msg = "This method is only available when using the Gurobi solver."
        with pytest.raises(NotImplementedError, match=error_msg):
            runner.invoke(
                cli_run, ["--config", str(config_ini_path)], catch_exceptions=False
            )
            assert not gurobi_parameter_path.exists()


def test_simple_run_with_generator_capacity_cost(
    config_ini_path: Path,
    config_parser_with_creator: configparser.ConfigParser,
) -> None:
    config = config_parser_with_creator
    config["optimization"]["generator_capacity_cost"] = "netto"
    set_up_config_ini(config_ini_path, config)
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


def test_simple_run_and_feather_dump(
    config_ini_path: Path,
    config_parser_with_feather: configparser.ConfigParser,
    output_path: Path,
    csv_dump_path: Path,
) -> None:
    set_up_config_ini(config_ini_path, config_parser_with_feather)
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
        output_path / "feather" / "Objective_func_value.feather"
    ).is_file()
    assert (
        set(x.name for x in (output_path / "csv").iterdir() if x.is_dir())
        == results_expected_dirs
    )
    assert (
        set(x.name for x in (output_path / "feather").iterdir() if x.is_dir())
        == results_expected_dirs
    )


@pytest.mark.parametrize(
    "hash_commit_flag",
    [
        pytest.param(True, id="dump_commit_on"),
        pytest.param(False, id="dump_commit_off"),
    ],
)
def test_simple_run_with_dump_git_info(
    config_ini_path: Path,
    config_parser: configparser.ConfigParser,
    output_path: Path,
    hash_commit_flag: bool,
) -> None:
    set_up_config_ini(config_ini_path, config_parser)
    runner = CliRunner()

    if hash_commit_flag:
        runner.invoke(
            cli_run, ["--config", str(config_ini_path), "-hcd"], catch_exceptions=False
        )
        assert (output_path / "git_info.txt").exists()
        assert (output_path / "git_info.txt").is_file()
    else:
        runner.invoke(
            cli_run, ["--config", str(config_ini_path)], catch_exceptions=False
        )
        assert not (output_path / "git_info.txt").exists()
        assert not (output_path / "git_info.txt").is_file()
