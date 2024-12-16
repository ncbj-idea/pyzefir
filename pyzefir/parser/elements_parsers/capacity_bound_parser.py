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
    """
    Parses and processes data to create instances of CapacityBound.

    This class takes a DataFrame that contains information about capacity
    bounds and processes it to generate CapacityBound instances. It ensures
    that the attributes, such as technology names, coefficients, and sense,
    are properly extracted and formatted from the input data.
    """

    def __init__(
        self,
        capacity_bound_df: pd.DataFrame,
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - capacity_bound_df (pd.DataFrame): DataFrame containing capacity bound information for parsing.
        """
        self.capacity_bound_df = capacity_bound_df

    def create(self) -> tuple[CapacityBound, ...]:
        """
        Creates and returns a tuple of CapacityBound instances.

        This method processes the capacity bound DataFrame, ensuring the
        "sense" column is capitalized and handles missing coefficient values
        by assigning them a default value of 1.0. It converts each row into a
        CapacityBound object.

        Returns:
            - tuple[CapacityBound, ...]: A tuple containing the created CapacityBound instances.
        """
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
