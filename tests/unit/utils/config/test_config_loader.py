import re
from pathlib import Path

import numpy as np
import pytest

from pyzefir.utils.config_parser import ConfigException, ConfigLoader
from tests.unit.utils.config.utils import (
    create_test_config_file,
    dump_test_config_file,
    dump_vector_data_from_csv,
)


def test_valid_minimal_config_file(
    tmp_path: Path, mock_input_directory: Path, mock_output_directory: Path
) -> None:
    """Test if config file will be correctly loaded, if only required parameters will be provided."""
    config_file = create_test_config_file(
        input_dict={
            "input_path": str(mock_input_directory),
            "scenario": "scenario",
            "input_format": "csv",
        },
        output_dict={"output_path": str(mock_output_directory)},
    )
    dump_test_config_file(config_file, tmp_path / "config.ini")
    loaded_params = ConfigLoader(tmp_path / "config.ini").load()

    assert loaded_params.input_path == mock_input_directory
    assert loaded_params.output_path == mock_output_directory
    assert loaded_params.scenario == "scenario"
    assert loaded_params.input_format == "csv"
    assert loaded_params.sol_dump_path == mock_output_directory / "results.sol"
    assert loaded_params.opt_logs_path == mock_output_directory / "opt.log"
    assert loaded_params.csv_dump_path is None
    assert loaded_params.hour_sample is None
    assert loaded_params.year_sample is None
    assert loaded_params.discount_rate is None


@pytest.mark.parametrize(
    ("input_path", "output_path", "error_msg"),
    [
        (
            "invalid_path",
            None,
            "Path specified as input_path should exist: ",
        ),
    ],
)
def test_incorrect_required_params(
    input_path: str | None,
    output_path: str | None,
    error_msg: str,
    tmp_path: Path,
    mock_output_directory: Path,
    mock_input_directory: Path,
) -> None:
    """Test if for invalid input / output paths error with appropriate message will be raised."""
    config_file = create_test_config_file(
        input_dict={
            "input_path": str(tmp_path / (input_path or str(mock_input_directory))),
            "scenario": "scenario",
            "input_format": "csv",
        },
        output_dict={
            "output_path": str(tmp_path / (output_path or str(mock_output_directory)))
        },
    )
    dump_test_config_file(config_file, tmp_path / "config.ini")
    with pytest.raises(ConfigException, match=error_msg):
        ConfigLoader(tmp_path / "config.ini").load()


def test_complete_config_file(
    tmp_path: Path,
    mock_input_directory: Path,
    mock_output_directory: Path,
    mock_tmp_dir: Path,
) -> None:
    """Test if all parameters are correct, all are loaded correctly."""
    config_file = create_test_config_file(
        input_dict={
            "input_path": str(mock_input_directory),
            "scenario": "scenario",
            "input_format": "xlsx",
        },
        output_dict={
            "output_path": str(mock_output_directory),
            "sol_dump_path": str(mock_tmp_dir / "model.sol"),
            "csv_dump_path": str(mock_tmp_dir),
            "opt_logs_path": str(mock_tmp_dir / "optimization.log"),
        },
        parameters_dict={
            "year_sample": str(tmp_path / "year_sample.csv"),
            "discount_rate": str(tmp_path / "discount_rate.csv"),
            "hour_sample": str(tmp_path / "hour_sample.csv"),
        },
    )
    dump_test_config_file(config_file, tmp_path / "config.ini")
    dump_vector_data_from_csv(np.arange(100), tmp_path / "hour_sample.csv")
    dump_vector_data_from_csv(np.arange(5), tmp_path / "year_sample.csv")
    dump_vector_data_from_csv(
        np.array([0.05, 0.07, 0.1, 0.06, 0.03]), tmp_path / "discount_rate.csv"
    )
    loaded_params = ConfigLoader(tmp_path / "config.ini").load()

    assert loaded_params.input_path == mock_input_directory
    assert loaded_params.scenario == "scenario"
    assert loaded_params.input_format == "xlsx"
    assert loaded_params.csv_dump_path == mock_tmp_dir
    assert loaded_params.output_path == mock_output_directory
    assert loaded_params.sol_dump_path == mock_tmp_dir / "model.sol"
    assert loaded_params.opt_logs_path == mock_tmp_dir / "optimization.log"
    assert np.all(loaded_params.year_sample == np.arange(5))
    assert np.all(loaded_params.hour_sample == np.arange(100))
    assert np.all(loaded_params.discount_rate == [0.05, 0.07, 0.1, 0.06, 0.03])


def test_valid_feather_input_format(
    tmp_path: Path, mock_input_directory: Path, mock_output_directory: Path
) -> None:
    config_file = create_test_config_file(
        input_dict={
            "input_path": str(mock_input_directory),
            "scenario": "scenario",
            "input_format": "feather",
        },
        output_dict={"output_path": str(mock_output_directory)},
    )
    dump_test_config_file(config_file, tmp_path / "config.ini")
    config = ConfigLoader(tmp_path / "config.ini").load()
    assert config.input_format == "feather"


@pytest.mark.parametrize(
    ("hour_sample", "year_sample", "discount_rate", "error_msg"),
    [
        ("", "", "your", "Path specified as discount_rate does not exist: your"),
        ("", "mother", "", "Path specified as year_sample does not exist: mother"),
        (
            "washes in the river",
            "",
            "",
            "Path specified as hour_sample does not exist: washes in the river",
        ),
    ],
)
def test_invalid_optional_parameters(
    hour_sample: str | None,
    year_sample: str | None,
    discount_rate: str | None,
    error_msg: str,
    tmp_path: Path,
    mock_input_directory: Path,
    mock_output_directory: Path,
) -> None:
    config_file = create_test_config_file(
        input_dict={
            "input_path": str(mock_input_directory),
            "scenario": "scenario",
            "input_format": "csv",
        },
        output_dict={"output_path": str(mock_output_directory)},
        parameters_dict={
            "year_sample": year_sample,
            "discount_rate": discount_rate,
            "hour_sample": hour_sample,
        },
    )
    dump_test_config_file(config_file, tmp_path / "config.ini")
    with pytest.raises(ConfigException, match=error_msg):
        ConfigLoader(tmp_path / "config.ini").load()


@pytest.mark.parametrize(
    ("parameter_name", "data"),
    [
        ("hour_sample", np.arange(100).reshape(2, 50)),
        ("year_sample", np.arange(10).reshape((5, 2))),
        ("discount_rate", np.ones((2, 2))),
    ],
)
def test_invalid_parameters(
    parameter_name: str,
    data: np.ndarray,
    tmp_path: Path,
    mock_input_directory: Path,
    mock_output_directory: Path,
) -> None:
    config_file = create_test_config_file(
        input_dict={
            "input_path": str(mock_input_directory),
            "scenario": "scenario",
            "input_format": "csv",
        },
        output_dict={"output_path": str(mock_output_directory)},
        parameters_dict={parameter_name: str(tmp_path / "data.csv")},
    )
    dump_test_config_file(config_file, tmp_path / "config.ini")
    dump_vector_data_from_csv(data, tmp_path / "data.csv")
    expected_msg = f"provided {parameter_name} is {data.ndim} dimensional dataset, one dimensional data is required"
    with pytest.raises(ConfigException, match=expected_msg):
        ConfigLoader(tmp_path / "config.ini").load()


def test_invalid_input_format(
    tmp_path: Path, mock_input_directory: Path, mock_output_directory: Path
) -> None:
    config_file = create_test_config_file(
        input_dict={
            "input_path": str(mock_input_directory),
            "scenario": "scenario",
            "input_format": "aaa",
        },
        output_dict={"output_path": str(mock_output_directory)},
    )
    dump_test_config_file(config_file, tmp_path / "config.ini")
    with pytest.raises(
        ConfigException,
        match="provided input_format aaa is different than valid formats: csv, xlsx or feather",
    ):
        ConfigLoader(tmp_path / "config.ini").load()


def test_xlsx_input_format_without_csv_dump_path(
    tmp_path: Path, mock_input_directory: Path, mock_output_directory: Path
) -> None:
    config_file = create_test_config_file(
        input_dict={
            "input_path": str(mock_input_directory),
            "scenario": "scenario",
            "input_format": "xlsx",
        },
        output_dict={"output_path": str(mock_output_directory)},
    )
    dump_test_config_file(config_file, tmp_path / "config.ini")
    with pytest.raises(
        ConfigException, match="csv_dump_path should be specified for xlsx input_format"
    ):
        ConfigLoader(tmp_path / "config.ini").load()


def test_csv_input_format_with_csv_dump_path(
    tmp_path: Path,
    mock_input_directory: Path,
    mock_output_directory: Path,
    mock_tmp_dir: Path,
) -> None:
    config_file = create_test_config_file(
        input_dict={
            "input_path": str(mock_input_directory),
            "scenario": "scenario",
            "input_format": "csv",
        },
        output_dict={
            "output_path": str(mock_output_directory),
            "csv_dump_path": str(mock_tmp_dir),
        },
    )
    dump_test_config_file(config_file, tmp_path / "config.ini")
    with pytest.raises(
        ConfigException,
        match="csv_dump_path should not be specified for csv input_format",
    ):
        ConfigLoader(tmp_path / "config.ini").load()


def test_invalid_suffix_sol_dump_path(
    tmp_path: Path,
    mock_input_directory: Path,
    mock_output_directory: Path,
    mock_tmp_dir: Path,
) -> None:
    config_file = create_test_config_file(
        input_dict={
            "input_path": str(mock_input_directory),
            "scenario": "scenario",
            "input_format": "csv",
        },
        output_dict={
            "output_path": str(mock_output_directory),
            "sol_dump_path": tmp_path / Path("uga_uga") / "model.ini",
        },
    )
    dump_test_config_file(config_file, tmp_path / "config.ini")
    msg = "Path specified as sol_dump_path directory should exist: .*uga_uga"
    with pytest.raises(ConfigException, match=msg):
        ConfigLoader(tmp_path / "config.ini").load()


def test_invalid_directory_path_sol_dump_path(
    tmp_path: Path,
    mock_input_directory: Path,
    mock_output_directory: Path,
    mock_tmp_dir: Path,
) -> None:
    config_file = create_test_config_file(
        input_dict={
            "input_path": str(mock_input_directory),
            "scenario": "scenario",
            "input_format": "csv",
        },
        output_dict={
            "output_path": str(mock_output_directory),
            "sol_dump_path": tmp_path / "model.ini",
        },
    )
    dump_test_config_file(config_file, tmp_path / "config.ini")
    msg = re.escape(
        "Path specified as sol_dump_path has incorrect suffix: model.ini (expected .sol)"
    )
    with pytest.raises(ConfigException, match=msg):
        ConfigLoader(tmp_path / "config.ini").load()


@pytest.mark.parametrize(
    "param_value,is_correct",
    (("netto", True), ("brutto", True), ("bum-shakalaka!", False)),
)
def test_load_generator_capacity_cost_parameter(
    tmp_path: Path,
    mock_input_directory: Path,
    mock_output_directory: Path,
    mock_tmp_dir: Path,
    param_value: str,
    is_correct: bool,
) -> None:
    """
    Test if config file loading behaves correctly with generator capacity cost parameter.
    """
    config_file = create_test_config_file(
        input_dict={
            "input_path": str(mock_input_directory),
            "scenario": "scenario",
            "input_format": "csv",
        },
        output_dict={
            "output_path": str(mock_output_directory),
            "sol_dump_path": tmp_path / "model.sol",
        },
    )
    config_file.add_section("optimization")
    config_file["optimization"].update(generator_capacity_cost=param_value)
    dump_test_config_file(config_file, tmp_path / "config.ini")
    if is_correct:
        config = ConfigLoader(tmp_path / "config.ini").load()
        assert (
            config.network_config["generator_capacity_cost"] == param_value or "netto"
        )
    else:
        msg = re.escape(
            f"given value of a generator_capacity_cost {param_value} is different than allowed values "
            "netto or brutto"
        )
        with pytest.raises(ConfigException, match=msg):
            ConfigLoader(tmp_path / "config.ini").load()


@pytest.mark.parametrize(
    "value, expected_value",
    (
        ("", ""),
        ("-5", -5),
        ("abc", "abc"),
        ("+10", 10),
        ("1e-5", 1e-5),
        ("1e10", int(1e10)),  # 1e10 gives float, but we interpret it as int
        ("1.23", 1.23),
        ("1,23", "1,23"),
        ("True", True),
        ("false", False),
    ),
)
def test_try_parse_config_option(value: str, expected_value: str | float) -> None:
    parsed_value = ConfigLoader.try_parse_config_option(value)
    assert parsed_value == expected_value and type(parsed_value) is type(expected_value)
