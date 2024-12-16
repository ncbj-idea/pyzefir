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

from pyzefir.model.network_elements import GenerationFraction
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class GenerationFractionParserException(Exception):
    pass


class GenerationFractionParser(AbstractElementParser):
    """
    Parses generation fractions from a DataFrame to create GenerationFraction objects.

    This class processes generation fraction data over a specified number of years,
    validating unique values for different categories, and constructing GenerationFraction instances.
    It ensures that all relevant attributes for generation fractions are correctly represented.
    """

    def __init__(
        self,
        generation_fraction_df: pd.DataFrame,
        n_years: int,
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - generation_fraction_df (pd.DataFrame): DataFrame containing generation fraction data.
            - n_years (int): The number of years for which generation fractions are defined.
        """
        generation_fraction_df["type"] = generation_fraction_df["type"].str.lower()
        self.generation_fraction_df = generation_fraction_df
        self.n_years = n_years

    def create(self) -> tuple[GenerationFraction, ...]:
        """
        Creates GenerationFraction objects from the DataFrame.

        This method groups the generation fraction data by name and constructs
        GenerationFraction instances for each group. The result is a tuple of all created instances.

        Returns:
            - tuple[GenerationFraction, ...]: A tuple of GenerationFraction instances created from the input DataFrame.
        """
        generation_fractions: list[GenerationFraction] = []
        for name, df in self.generation_fraction_df.groupby("name"):
            generation_fraction = self._create_generation_fraction(name, df)
            generation_fractions.append(generation_fraction)
        return tuple(generation_fractions)

    def _create_generation_fraction(
        self, name: str, single_gf_df: pd.DataFrame
    ) -> GenerationFraction:
        """
        Creates a GenerationFraction object from a DataFrame segment.

        Args:
            - name (str): The name of the generation fraction.
            - single_gf_df (pd.DataFrame): A DataFrame segment corresponding to a single generation fraction.

        Returns:
            - GenerationFraction: An instance of the GenerationFraction class populated with the data.
        """
        unique_tag: np.ndarray = single_gf_df["tag"].unique()
        unique_sub_tag: np.ndarray = single_gf_df["subtag"].unique()
        unique_energy_type: np.ndarray = single_gf_df["energy_type"].unique()
        unique_fraction_type: np.ndarray = single_gf_df["type"].unique()
        self._validate_unique_values(
            name, (unique_energy_type, unique_fraction_type, unique_sub_tag, unique_tag)
        )
        min_generation_fraction = self._create_fraction_series(
            single_gf_df, "min_generation_fraction", self.n_years
        )
        max_generation_fraction = self._create_fraction_series(
            single_gf_df, "max_generation_fraction", self.n_years
        )
        return GenerationFraction(
            name=name,
            tag=unique_tag[0],
            sub_tag=unique_sub_tag[0],
            energy_type=unique_energy_type[0],
            fraction_type=unique_fraction_type[0],
            min_generation_fraction=min_generation_fraction,
            max_generation_fraction=max_generation_fraction,
        )

    @staticmethod
    def _validate_unique_values(
        name: str, unique_values: tuple[np.ndarray, ...]
    ) -> None:
        """
        Validates that only one unique value exists for each category.

        Args:
            - name (str): The name associated with the unique values.
            - unique_values (tuple[np.ndarray, ...]): A tuple of arrays containing unique values to validate.

        Raises:
            - GenerationFractionParserException: If more than one unique value is found for any category.
        """
        for unique_value in unique_values:
            if len(unique_value) > 1:
                raise GenerationFractionParserException(
                    f"In frame found more than one unique value: {unique_values} for name {name}"
                )

    @staticmethod
    def _create_fraction_series(
        df: pd.DataFrame, column_name: str, n_years: int
    ) -> pd.Series:
        """
        Creates a series of fractions from the DataFrame for a given column.

        Args:
            - df (pd.DataFrame): The DataFrame containing the data.
            - column_name (str): The name of the column from which to create the fraction series.
            - n_years (int): The number of years for which the series should be created.

        Returns:
            - pd.Series: A series of fractions reindexed to the specified number of years.
        """
        years = range(0, n_years)
        pivot_df = df.pivot_table(
            index="year",
            values=column_name,
        )
        if pivot_df.empty:
            return pd.Series(index=years)
        fraction_series = pivot_df[column_name].reindex(years)
        return fraction_series
