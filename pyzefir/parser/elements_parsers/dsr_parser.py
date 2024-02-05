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

import pandas as pd

from pyzefir.model.network_elements.dsr import DSR
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class DSRParser(AbstractElementParser):
    def __init__(
        self,
        dsr_df: pd.DataFrame,
    ) -> None:
        self.dsr_df = dsr_df

    def create(self) -> tuple[DSR, ...]:
        return tuple(
            DSR(
                name=row["name"],
                compensation_factor=row["compensation_factor"],
                balancing_period_len=row["balancing_period_len"],
                penalization=row["penalization"],
                relative_shift_limit=row["relative_shift_limit"]
                if not pd.isnull(row["relative_shift_limit"])
                else None,
                abs_shift_limit=row["abs_shift_limit"]
                if not pd.isnull(row["abs_shift_limit"])
                else None,
            )
            for row in self.dsr_df.to_dict(orient="records")
        )
