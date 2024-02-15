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

import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from pyzefir.structure_creator.cli.cli_wrapper import run_structure_creator_cli
from tests.utils import RESOURCES_PATH


@pytest.fixture
def output_path() -> Path:
    """Temporary output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield str(temp_dir)


def test_main(output_path: str) -> None:
    runner = CliRunner()
    runner.invoke(
        run_structure_creator_cli,
        args=[
            "--input_path",
            Path(RESOURCES_PATH) / "structure_creator_resources",
            "--output_path",
            output_path,
            "--scenario_name",
            "base_variant",
            "--n_hours",
            "100",
            "--n_years",
            "2",
        ],
        catch_exceptions=False,
    )
