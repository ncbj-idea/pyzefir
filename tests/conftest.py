from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.utils import get_resources

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


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


@pytest.fixture
def temp_working_directory_mock(
    monkeypatch: MonkeyPatch, temporary_directory: Path
) -> None:
    monkeypatch.chdir(temporary_directory)
