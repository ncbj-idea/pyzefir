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
from pyzefir.parser.elements_parsers.utils import (
    convert_to_float,
    numeric_value_or_default,
)


class DSRParser(AbstractElementParser):
    """
    Parses and processes data to create instances of DSR.

    This class takes a DataFrame containing demand-side response (DSR) data and processes
    it to generate DSR instances. Each DSR object is created from the input data, including
    attributes such as compensation factor, balancing period length, and various limits.
    """

    def __init__(
        self,
        dsr_df: pd.DataFrame,
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - dsr_df (pd.DataFrame): DataFrame containing demand-side response data for parsing.
        """
        self.dsr_df = dsr_df

    def create(self) -> tuple[DSR, ...]:
        """
        Creates and returns a tuple of DSR instances.

        This method processes the DSR DataFrame to create DSR objects, extracting and converting
        the necessary attributes from each row of the DataFrame. It handles missing values for
        the relative and absolute shift limits by setting them to None when appropriate.

        Returns:
            - tuple[DSR, ...]: A tuple containing the created DSR instances.
        """
        return tuple(
            DSR(
                name=str(row["name"]),
                compensation_factor=float(row["compensation_factor"]),
                balancing_period_len=int(row["balancing_period_len"]),
                penalization_minus=numeric_value_or_default(
                    float(row["penalization_minus"]), default_value=0.0
                ),
                penalization_plus=numeric_value_or_default(
                    float(row["penalization_plus"]), default_value=0.0
                ),
                relative_shift_limit=convert_to_float(row["relative_shift_limit"]),
                abs_shift_limit=convert_to_float(row["abs_shift_limit"]),
                hourly_relative_shift_minus_limit=numeric_value_or_default(
                    row["hourly_relative_shift_minus_limit"],
                    default_value=1.0,
                ),
                hourly_relative_shift_plus_limit=numeric_value_or_default(
                    row["hourly_relative_shift_plus_limit"],
                    default_value=1.0,
                ),
            )
            for row in self.dsr_df.to_dict(orient="records")
        )
