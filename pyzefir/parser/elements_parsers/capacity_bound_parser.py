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

import numpy as np
import pandas as pd

from pyzefir.model.network_elements import CapacityBound
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class CapacityBoundParser(AbstractElementParser):
    def __init__(
        self,
        capacity_bound_df: pd.DataFrame,
    ) -> None:
        self.capacity_bound_df = capacity_bound_df

    def create(self) -> tuple[CapacityBound, ...]:
        self.capacity_bound_df.loc[:, "sense"] = self.capacity_bound_df[
            "sense"
        ].str.upper()
        return tuple(
            CapacityBound(
                name=row["name"],
                left_technology=row["left_technology_name"],
                right_technology=row["right_technology_name"],
                sense=row["sense"],
                left_coefficient=(
                    1.0 if np.isnan(row["left_coeff"]) else float(row["left_coeff"])
                ),
            )
            for row in self.capacity_bound_df.to_dict(orient="records")
        )
