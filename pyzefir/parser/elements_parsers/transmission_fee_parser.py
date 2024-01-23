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

from pyzefir.model.network_elements import TransmissionFee
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class TransmissionFeeParser(AbstractElementParser):
    def __init__(self, transmission_fee_df: pd.DataFrame) -> None:
        self.transmission_fee_df = transmission_fee_df

    def create(self) -> tuple[TransmissionFee, ...]:
        self.transmission_fee_df.set_index("hour_idx", inplace=True)

        return tuple(
            TransmissionFee(name=name, fee=self.transmission_fee_df[name])
            for name in self.transmission_fee_df.columns
        )
