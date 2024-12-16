# PyZefir
# Copyright (C) 2024 Narodowe Centrum Badań Jądrowych
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

import os
from pathlib import Path

import pytest

from pyzefir import ROOT_DIR
from pyzefir.utils.git_info_dumper import GitInfoDumper


@pytest.fixture
def git_info() -> GitInfoDumper:
    return GitInfoDumper(ROOT_DIR.parent.parent)


def test_dump_git_info(git_info: GitInfoDumper, tmp_path: Path) -> None:
    git_info.dump_git_info(tmp_path)
    file_path = tmp_path / "git_info.txt"
    assert file_path.exists()
    assert os.stat(file_path).st_size > 0


def test_invalid_repo_path(tmp_path: Path) -> None:
    git_info = GitInfoDumper(Path("invalid/path"))
    assert git_info.repo is None

    git_info.dump_git_info(tmp_path)
    file_path = tmp_path / "git_info.txt"
    assert not file_path.exists()
