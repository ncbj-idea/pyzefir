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
