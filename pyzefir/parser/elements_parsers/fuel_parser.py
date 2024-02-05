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

from pyzefir.model.network_elements import Fuel
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser
from pyzefir.utils.path_manager import DataSubCategories


class FuelParser(AbstractElementParser):
    def __init__(
        self,
        emission_per_unit_df: pd.DataFrame,
        energy_per_unit_df: pd.DataFrame,
        fuel_prices_df: pd.DataFrame,
        fuel_availability_df: pd.DataFrame,
    ) -> None:
        self.fuel_availability_df = fuel_availability_df
        self.fuel_prices_df = fuel_prices_df
        self.energy_per_unit_df = energy_per_unit_df
        self.emission_per_unit_df = emission_per_unit_df

    def create(self) -> tuple[Fuel, ...]:
        energy_per_unit_df = self.energy_per_unit_df.copy(deep=True)
        emission_per_unit_df = self.emission_per_unit_df.copy(deep=True)
        fuel_availability = self.fuel_availability_df.copy(deep=True)
        fuel_prices = self.fuel_prices_df.copy(deep=True)

        fuel_df = FuelParser._merge_energy_emission_data(
            energy_per_unit_df, emission_per_unit_df
        )

        return tuple(
            fuel_df.apply(
                FuelParser._create_fuel, axis=1, args=(fuel_prices, fuel_availability)
            )
        )

    @staticmethod
    def _create_fuel(
        df_row: pd.Series, fuel_prices: pd.DataFrame, fuel_availability: pd.DataFrame
    ) -> Fuel:
        return Fuel(
            name=str(df_row.name),
            emission={
                emission_type: value for emission_type, value in df_row[1:].items()
            },
            availability=(
                fuel_availability[df_row.name]
                if df_row.name in fuel_availability.columns
                else None
            ),
            cost=fuel_prices[df_row.name],
            energy_per_unit=df_row["energy_per_unit"],
        )

    @staticmethod
    def _merge_energy_emission_data(
        energy_per_unit_df: pd.DataFrame, emission_per_unit_df: pd.DataFrame
    ) -> pd.DataFrame:
        energy_per_unit_df = energy_per_unit_df.set_index("name", drop=True)
        emission_per_unit_df = emission_per_unit_df.set_index("name", drop=True)

        fuel_df = pd.concat([energy_per_unit_df, emission_per_unit_df], axis=1)
        if fuel_df.isnull().values.any():
            raise ValueError(
                f"Fuel names in {DataSubCategories.EMISSION_PER_UNIT} must "
                f"match with {DataSubCategories.ENERGY_PER_UNIT}"
            )

        return fuel_df
