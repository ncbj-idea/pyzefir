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

from pyzefir.model.network_elements.emission_fee import EmissionFee
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class EmissionFeeParser(AbstractElementParser):
    def __init__(
        self,
        emission_type_df: pd.DataFrame,
        emission_fee_df: pd.DataFrame,
    ) -> None:
        self.emission_fee_df = emission_fee_df.set_index("year_idx")
        self.emission_type_df = emission_type_df.set_index("emission_fee").squeeze()

    def create(self) -> tuple[EmissionFee, ...]:
        return tuple(
            EmissionFee(
                name=name,
                price=self.emission_fee_df[name].astype(float),
                emission_type=self.emission_type_df[name],
            )
            for name in self.emission_fee_df.columns
        )
