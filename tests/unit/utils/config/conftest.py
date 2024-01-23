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

import pytest


@pytest.fixture
def mock_input_directory() -> Path:
    """Fixture to create a mock input directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_output_directory() -> Path:
    """Fixture to create a mock output directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_tmp_file(tmp_path: Path) -> Path:
    """Fixture to create temporary file for testing."""
    tmp_file_path = tmp_path / "tmp_file"
    tmp_file_path.touch()
    return tmp_file_path


@pytest.fixture
def mock_tmp_dir(tmp_path: Path) -> Path:
    """Fixture to create temporary directory for testing."""
    tmp_dir_path = tmp_path / "tmp_dir"
    tmp_dir_path.mkdir(exist_ok=True)
    yield tmp_dir_path
    shutil.rmtree(tmp_dir_path)  # Cleanup: Remove the mock directory after the test
