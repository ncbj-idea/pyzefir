import tempfile
from pathlib import Path

import pytest

from tests.utils import get_resources


@pytest.fixture
def csv_root_path() -> Path:
    return get_resources("parser_input_csv")


@pytest.fixture
def test_network_assets_path() -> Path:
    return get_resources("test_network_assets")


@pytest.fixture
def temporary_directory() -> Path:
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)
