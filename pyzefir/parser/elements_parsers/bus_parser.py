import pandas as pd

from pyzefir.model.network_elements import Bus
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class BusParser(AbstractElementParser):
    """
    Parses and processes data to create instances of Bus.

    This class takes a DataFrame containing information about different buses
    and processes it to create Bus instances. It ensures that each bus's attributes,
    such as name, energy type, and DSR type, are properly set from the input data.
    """

    def __init__(
        self,
        bus_df: pd.DataFrame,
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - bus_df (pd.DataFrame): DataFrame containing information about the buses to be parsed.
        """
        self.bus_df = bus_df

    def create(self) -> tuple[Bus, ...]:
        """
        Creates and returns a tuple of Bus instances.

        This method processes the bus DataFrame and converts each row into a Bus object.
        It handles missing values for the DSR type by setting them to None when appropriate.

        Returns:
            - tuple[Bus, ...]: A tuple containing the created Bus instances.
        """
        return tuple(
            Bus(
                name=str(row["name"]),
                energy_type=str(row["energy_type"]),
                dsr_type=(
                    str(row["dsr_type"]) if not pd.isnull(row["dsr_type"]) else None
                ),
            )
            for row in self.bus_df.to_dict(orient="records")
        )
