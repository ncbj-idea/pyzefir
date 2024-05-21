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


import logging
from pathlib import Path

import pandas as pd

_logger = logging.getLogger(__name__)


def write_to_excel(
    data: dict,
    output_path: Path,
    filename: str,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / filename

    with pd.ExcelWriter(
        path=output_path,
        engine="openpyxl",
        mode="a" if output_path.is_file() else "w",
        if_sheet_exists="replace" if output_path.is_file() else None,
    ) as writer:
        for sheet_name, sheet_data in data.items():
            sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
            _logger.debug("Data written to sheet %s in file %s", sheet_name, filename)
        _logger.info("File %s has been saved at %s", filename, output_path)
