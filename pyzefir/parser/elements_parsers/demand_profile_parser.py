import pandas as pd

from pyzefir.model.network_elements import DemandProfile
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class DemandProfileParser(AbstractElementParser):
    """
    Parses and processes data to create instances of DemandProfile.

    This class takes a dictionary of DataFrames containing demand data and processes
    it to generate DemandProfile instances. Each demand profile is created by normalizing
    the data and organizing it into a structured format that associates the profile name
    with its corresponding demand values.
    """

    def __init__(
        self,
        demand_dict: dict[str, pd.DataFrame],
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - demand_dict (dict[str, pd.DataFrame]): A dictionary mapping demand profile names
                to their corresponding DataFrames.
        """
        self.demand_dict = demand_dict

    def create(self) -> tuple[DemandProfile, ...]:
        """
        Creates and returns a tuple of DemandProfile instances.

        This method processes each DataFrame in the demand dictionary to create DemandProfile
        objects. It sets the index of each demand DataFrame to "hour_idx" and converts the
        demand data into a dictionary format that associates hours with normalized demand values.

        Returns:
            - tuple[DemandProfile, ...]: A tuple containing the created DemandProfile instances.
        """
        demand_profiles: list[DemandProfile] = list()
        for name, demand_df in self.demand_dict.items():
            demand_profile = DemandProfile(
                name=str(name),
                normalized_profile=demand_df.set_index("hour_idx").to_dict("series"),
            )
            demand_profiles.append(demand_profile)

        return tuple(demand_profiles)
