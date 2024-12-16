import pandas as pd

from pyzefir.model.network_elements import Fuel
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser
from pyzefir.utils.path_manager import DataSubCategories


class FuelParser(AbstractElementParser):
    """
    Parses fuel-related data from various DataFrames to create Fuel objects.

    This class consolidates information about emissions, energy content, availability, and pricing of fuels.
    It facilitates the creation of Fuel instances for use in energy simulations and analyses.
    """

    def __init__(
        self,
        emission_per_unit_df: pd.DataFrame,
        energy_per_unit_df: pd.DataFrame,
        fuel_prices_df: pd.DataFrame,
        fuel_availability_df: pd.DataFrame,
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - emission_per_unit_df (pd.DataFrame): DataFrame containing emission data per unit of fuel.
            - energy_per_unit_df (pd.DataFrame): DataFrame containing energy data per unit of fuel.
            - fuel_prices_df (pd.DataFrame): DataFrame containing price data for fuels.
            - fuel_availability_df (pd.DataFrame): DataFrame containing availability information for fuels.
        """
        self.fuel_availability_df = fuel_availability_df
        self.fuel_prices_df = fuel_prices_df
        self.energy_per_unit_df = energy_per_unit_df
        self.emission_per_unit_df = emission_per_unit_df

    def create(self) -> tuple[Fuel, ...]:
        """
        Creates Fuel objects from the DataFrames provided.

        This method merges energy and emission data into a single DataFrame and then
        creates Fuel instances for each entry. The resulting tuple contains all Fuel objects
        constructed from the data.

        Returns:
            - tuple[Fuel, ...]: A tuple of Fuel instances created from the input DataFrames.
        """
        energy_per_unit_df = self.energy_per_unit_df.copy(deep=True)
        emission_per_unit_df = self.emission_per_unit_df.copy(deep=True)
        fuel_availability = self.fuel_availability_df.copy(deep=True)
        fuel_prices = self.fuel_prices_df.copy(deep=True)

        fuel_df = FuelParser._merge_energy_emission_data(
            energy_per_unit_df, emission_per_unit_df
        )

        return tuple(
            fuel_df.apply(
                FuelParser._create_fuel,
                axis=1,
                args=(fuel_prices, fuel_availability),
                result_type="reduce",
            )
        )

    @staticmethod
    def _create_fuel(
        df_row: pd.Series, fuel_prices: pd.DataFrame, fuel_availability: pd.DataFrame
    ) -> Fuel:
        """
        Creates a Fuel object from a DataFrame row.

        Args:
            - df_row (pd.Series): A row of data representing a fuel.
            - fuel_prices (pd.DataFrame): DataFrame containing price information for fuels.
            - fuel_availability (pd.DataFrame): DataFrame containing availability information for fuels.

        Returns:
            - Fuel: An instance of the Fuel class populated with the row data.
        """
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
            energy_per_unit=float(df_row["energy_per_unit"]),
        )

    @staticmethod
    def _merge_energy_emission_data(
        energy_per_unit_df: pd.DataFrame, emission_per_unit_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges energy and emission DataFrames into a single DataFrame.

        This method sets the index for both DataFrames and concatenates them along the columns.
        It raises a ValueError if any fuel names do not match between the two DataFrames.

        Args:
            - energy_per_unit_df (pd.DataFrame): DataFrame containing energy data per unit of fuel.
            - emission_per_unit_df (pd.DataFrame): DataFrame containing emission data per unit of fuel.

        Returns:
            - pd.DataFrame: A merged DataFrame containing both energy and emission data.
        """
        energy_per_unit_df = energy_per_unit_df.set_index("name", drop=True)
        emission_per_unit_df = emission_per_unit_df.set_index("name", drop=True)

        fuel_df = pd.concat([energy_per_unit_df, emission_per_unit_df], axis=1)
        if fuel_df.isnull().values.any():
            raise ValueError(
                f"Fuel names in {DataSubCategories.EMISSION_PER_UNIT} must "
                f"match with {DataSubCategories.ENERGY_PER_UNIT}"
            )

        return fuel_df
