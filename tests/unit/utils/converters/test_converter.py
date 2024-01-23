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

from pyzefir.utils.converters.converter import AbstractConverter


def test_abstract_converter_convert() -> None:
    class MyConverter(AbstractConverter):
        def __init__(self) -> None:
            self.status = ""

        def convert(self) -> None:
            self.status = "Conversion complete"

    converter = MyConverter()
    converter.convert()
    assert converter.status == "Conversion complete"


def test_manage_existence_path_file_exists() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        path = tmp_dir_path / "output" / "test_file.txt"
        AbstractConverter.manage_existence_path(path)
        assert tmp_dir_path.joinpath("output").is_dir()
        assert not tmp_dir_path.joinpath("input").is_dir()
        assert not path.is_file()
