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
import logging
from pathlib import Path

from git import Repo

import pyzefir

_logger = logging.getLogger(__name__)


class GitInfoDumperWarning(Warning):
    pass


class GitInfoDumper:
    """
    Class for dumping Git repository information.

    This class provides functionality to extract and save information from a Git repository,
    such as the current commit SHA, author, and the Git diff.
    It initializes with a specified repository path and provides methods to dump this information into a text file.
    """

    def __init__(self, repo_path: Path) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - repo_path (Path): The path to the Git repository.
        """
        self.repo = self._setup_repo(repo_path)

    def _setup_repo(self, path: Path) -> Repo | None:
        """
        Set up the Git repository.

        Args:
            - path (Path): The path to the Git repository.

        Returns:
            - Repo | None: A Repo object if the path exists; None otherwise.
        """
        if not path.exists():
            _logger.warning(GitInfoDumperWarning(f"Path {path} doest not exist."))
            return None
        else:
            return Repo(path)

    def dump_git_info(self, path: Path, file_name: str = "git_info.txt") -> None:
        """
        Dump Git repository information into a text file.

        This method creates a text file containing the Pyzefir version, the current commit SHA,
        the commit author's name, and the Git diff. If the repository is not initialized correctly,
        a warning is logged instead.

        Args:
            - path (Path): The directory path where the Git information file will be saved.
            - file_name (str, optional): The name of the file to save the Git information. Defaults to "git_info.txt".
        """
        if self.repo is not None:
            txt = (
                f"Pyzefir version: {pyzefir.__version__}"
                f"Commit SHA: {self.repo.head.commit.hexsha},\n"
                f"Commit author: {self.repo.head.commit.committer.name},\n"
                f"Git diff: \n{self.repo.git.diff()}\n"
            )
            with open(path / file_name, "w", encoding="utf-8") as git_file:
                git_file.write(txt)
        else:
            _logger.warning(
                GitInfoDumperWarning(
                    "GitInfoDumper does not initialized because of invalid path. "
                    "File with git information does not saved."
                ),
            )
