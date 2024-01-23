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

import pandas as pd

from pyzefir.model.network_elements import Line
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class LineParser(AbstractElementParser):
    def __init__(self, line_df: pd.DataFrame) -> None:
        self.line_df = line_df

    def create(self) -> tuple[Line, ...]:
        lines = self.line_df.apply(
            self._create_line,
            axis=1,
        )
        return tuple(lines)

    @staticmethod
    def _create_line(df_row: pd.Series) -> Line:
        return Line(
            name=df_row["name"],
            energy_type=df_row["energy_type"],
            fr=df_row["bus_from"],
            to=df_row["bus_to"],
            transmission_loss=df_row["transmission_loss"],
            max_capacity=df_row["max_capacity"],
            transmission_fee=None
            if pd.isnull(df_row["transmission_fee"])
            else df_row["transmission_fee"],
        )
