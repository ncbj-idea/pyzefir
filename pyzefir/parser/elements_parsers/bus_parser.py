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

from pyzefir.model.network_elements import Bus
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class BusParser(AbstractElementParser):
    def __init__(
        self,
        bus_df: pd.DataFrame,
    ) -> None:
        self.bus_df = bus_df

    def create(self) -> tuple[Bus, ...]:
        return tuple(
            Bus(
                name=row["name"],
                energy_type=row["energy_type"],
                dsr_type=row["dsr_type"] if not pd.isnull(row["dsr_type"]) else None,
            )
            for row in self.bus_df.to_dict(orient="records")
        )
