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

import re
from pathlib import Path

import numpy as np
import pytest

from pyzefir.utils.config_parser import (
    ConfigException,
    load_vector_from_csv,
    validate_1D_array,
    validate_config_path,
    validate_csv_dump_path,
    validate_dir_path,
    validate_file_path,
    validate_input_format,
    validate_sol_dump_path,
)
from tests.unit.utils.config.utils import dump_vector_data_from_csv


def test_validate_config_path_incorrect_suffix(mock_tmp_file: Path) -> None:
    """Test if validate_config_path raises ConfigException when given path points to a file with incorrect extension."""
    msg = re.escape(
        f"Path specified as config_file_path has incorrect suffix: {mock_tmp_file.name} (expected .ini)"
    )
    with pytest.raises(ConfigException, match=msg):
        validate_config_path(mock_tmp_file)


def test_validate_config_path_incorrect_path() -> None:
    """Test if validate_config_path raises ConfigException when given path points to non-existing file."""
    msg = re.escape("Path specified as config_file_path does not exist: blabla")
    with pytest.raises(ConfigException, match=msg):
        validate_config_path(Path("blabla"))


def test_validate_config_path_correct_input(mock_tmp_dir: Path) -> None:
    """Test if validate_config_path will not raise ConfigException for correct input."""
    (mock_tmp_dir / "config.ini").touch()
    try:
        validate_config_path(mock_tmp_dir / "config.ini")
    except ConfigException:
        pytest.fail()


@pytest.mark.parametrize("file", ["file", "file.txt", "model.ini"])
def test_validate_sol_dump_path_incorrect_suffix(file: str, mock_tmp_dir: Path) -> None:
    """Test if validate_sol_dump_path raises ConfigException for path to a file with incorrect suffix."""
    msg = re.escape(
        f"Path specified as sol_dump_path has incorrect suffix: {file} (expected .sol)"
    )
    with pytest.raises(ConfigException, match=msg):
        validate_sol_dump_path(mock_tmp_dir / file)


@pytest.mark.parametrize(
    "path", [Path("dir/file.sol"), Path("your_mother / model.sol")]
)
def test_validate_sol_dump_path_incorrect_parent_directory(
    path: Path, tmp_path: Path
) -> None:
    """Test if validate_sol_dump_path raises ConfigException for path to a file in non-existing dir."""
    path = tmp_path / path
    msg = re.escape(
        f"Path specified as sol_dump_path directory should exist: {path.parent}"
    )
    with pytest.raises(ConfigException, match=msg):
        validate_sol_dump_path(path)


def test_validate_sol_dump_path_correct_input(mock_tmp_dir: Path) -> None:
    """Test if validate_sol_dump_path will not raise ConfigException for correct sol_file_path."""
    try:
        validate_sol_dump_path(mock_tmp_dir / "model.sol")
    except ConfigException:
        pytest.fail()


def test_validate_file_path_not_existing_path() -> None:
    """Test if validate_file_path raises ConfigException when the specified path does not exist."""
    with pytest.raises(
        ConfigException,
        match="Path specified as test_param does not exist: non_existent_file.txt",
    ):
        validate_file_path(Path("non_existent_file.txt"), param_name="test_param")


def test_validate_file_path_path_to_folder(mock_tmp_dir: Path) -> None:
    """Test if validate_file_path raises ConfigException when the specified path is not a dir path."""
    msg = re.escape(f"Path specified as tp does not point to a file: {mock_tmp_dir}")
    with pytest.raises(ConfigException, match=msg):
        validate_file_path(mock_tmp_dir, param_name="tp")


def test_validate_dir_path_not_existing_path() -> None:
    """Test if validate_dir_path raises ConfigException when the specified path is not a folder."""
    with pytest.raises(
        ConfigException,
        match="Path specified as test_param should exist: non_existent_folder",
    ):
        validate_dir_path(Path("non_existent_folder"), param_name="test_param")


def test_validate_dir_path_path_to_a_file(mock_tmp_file: Path) -> None:
    """Test if validate_dir_path raises ConfigException when the specified path is a file."""
    msg = re.escape(f"Path specified as p should point to a folder: {mock_tmp_file}")
    with pytest.raises(ConfigException, match=msg):
        validate_dir_path(mock_tmp_file, param_name="p")


def test_validate_data_vector() -> None:
    """Test if validate_data_vector will raise ConfigException for 2D NumPy array."""
    msg = re.escape(
        "provided param is 2 dimensional dataset, one dimensional data is required"
    )
    with pytest.raises(ConfigException, match=msg):
        validate_1D_array(np.zeros((1, 1)), "param")


@pytest.mark.parametrize("input_format", ("aaa", "CSV", ""))
def test_validate_input_format_invalid_data(input_format: str) -> None:
    """Test if validate_input_format raises ConfigException when invalid data is provided."""
    msg = f"provided input_format {input_format} is different than valid formats: csv, xlsx"
    with pytest.raises(ConfigException, match=msg):
        validate_input_format(input_format)


@pytest.mark.parametrize("input_format", ("csv", "xlsx"))
def test_validate_input_format(input_format: str) -> None:
    """Test if validate_input_format does not raise ConfigException for valid input."""
    try:
        validate_input_format(input_format)
    except ConfigException:
        pytest.fail()


@pytest.mark.parametrize(
    ("input_format", "csv_dump_path", "msg"),
    [
        ("xlsx", None, "csv_dump_path should be specified for xlsx input_format"),
        (
            "csv",
            Path("."),
            "csv_dump_path should not be specified for csv input_format",
        ),
    ],
)
def test_validate_csv_dump_path_invalid_parameters(
    input_format: str | None, csv_dump_path: Path | None, msg: str
) -> None:
    """Test if validate_csv_dump_path will not raise any error for valid input."""
    with pytest.raises(ConfigException, match=msg):
        validate_csv_dump_path(csv_dump_path, input_format)


@pytest.mark.parametrize(
    ("input_format", "csv_dump_path"), [("csv", None), ("xlsx", Path("."))]
)
def test_validate_csv_dump_path(
    input_format: str | None, csv_dump_path: Path | None
) -> None:
    """Test if validate_csv_dump_path will not raise ConfigException for invalid input."""
    try:
        validate_csv_dump_path(csv_dump_path, input_format)
    except ConfigException:
        pytest.fail()


def test_load_vector_data_from_csv_invalid_suffix(tmp_path: Path) -> None:
    """Test if load_vector_data_from_csv will raise ConfigException for invalid file suffix."""
    dump_vector_data_from_csv(np.arange(100), tmp_path / "hour_sample")
    msg = re.escape(
        "Path specified as param_name has incorrect suffix: hour_sample (expected .csv)"
    )
    with pytest.raises(ConfigException, match=msg):
        load_vector_from_csv(tmp_path / "hour_sample", "param_name")


def test_load_vector_from_csv_non_existing_file() -> None:
    """Test if load_vector_from_csv will raise ConfigException for path pointing to non-existing file."""
    with pytest.raises(
        ConfigException, match="Path specified as param_name does not exist: haha"
    ):
        load_vector_from_csv(Path("haha"), "param_name")


def test_load_vector_from_csv_directory(mock_tmp_dir: Path) -> None:
    """Test if load_vector_from_csv will raise ConfigException for path pointing to a directory."""
    msg = re.escape(f"Path specified as aaa does not point to a file: {mock_tmp_dir}")
    with pytest.raises(ConfigException, match=msg):
        load_vector_from_csv(mock_tmp_dir, "aaa")


@pytest.mark.parametrize("data", (np.arange(100), np.arange(10)))
def test_load_vector_from_csv(data: np.ndarray, tmp_path: Path) -> None:
    """Test if load_vector_from_csv will behave correctly with correct input data."""
    dump_vector_data_from_csv(data, tmp_path / "test_file.csv")
    arr = load_vector_from_csv(tmp_path / "test_file.csv", param_name="no_name")
    assert np.all(arr == arr)
