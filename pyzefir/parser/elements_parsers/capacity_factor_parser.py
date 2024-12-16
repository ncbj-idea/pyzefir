import pandas as pd

from pyzefir.model.network_elements import CapacityFactor
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class CapacityFactorParser(AbstractElementParser):
    """
    Parses and processes data to create instances of CapacityFactor.

    This class takes a DataFrame containing capacity factor data and processes
    it to generate CapacityFactor instances. Each column in the DataFrame represents
    a different capacity factor profile, and these are used to create CapacityFactor
    objects, which include the name and profile data for each factor.
    """

    def __init__(
        self,
        capacity_factors_df: pd.DataFrame,
    ):
        """
        Initializes a new instance of the class.

        Args:
            - capacity_factors_df (pd.DataFrame): DataFrame containing capacity factor profiles
              for various technologies.
        """
        self.capacity_factors_df = capacity_factors_df.copy(deep=True)

    def create(self) -> tuple[CapacityFactor, ...]:
        """
        Creates and returns a tuple of CapacityFactor instances.

        This method processes the capacity factor DataFrame by setting the index
        to "hour_idx" and extracting each column as a CapacityFactor object. Each
        column is used to create a CapacityFactor instance, where the column name
        serves as the name and the data as the profile.

        Returns:
            - tuple[CapacityFactor, ...]: A tuple containing the created CapacityFactor instances.
        """
        self.capacity_factors_df.set_index("hour_idx", inplace=True, drop=True)

        capacity_factors = []
        for col in self.capacity_factors_df.columns:
            capacity_factor = CapacityFactor(
                name=str(col), profile=self.capacity_factors_df[col]
            )
            capacity_factors.append(capacity_factor)
        return tuple(capacity_factors)
