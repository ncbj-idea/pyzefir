import pandas as pd

from pyzefir.model.network_elements import Line
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class LineParser(AbstractElementParser):
    """
    Parses transmission line data from a DataFrame to create Line objects.

    This class is responsible for converting a DataFrame containing transmission line data into
    a tuple of Line instances. Each line is characterized by attributes such as energy type,
    connected buses, transmission losses, capacities, and fees.
    """

    def __init__(self, line_df: pd.DataFrame) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - line_df (pd.DataFrame): DataFrame containing transmission line information.
        """
        self.line_df = line_df

    def create(self) -> tuple[Line, ...]:
        """
        Creates Line objects from the DataFrame.

        This method applies a function to each row of the DataFrame to create Line instances,
        returning a tuple of all created Line objects.

        Returns:
            - tuple[Line, ...]: A tuple of Line instances created from the input DataFrame.
        """
        lines = self.line_df.apply(
            self._create_line,
            axis=1,
        )
        return tuple(lines)

    @staticmethod
    def _create_line(df_row: pd.Series) -> Line:
        """
        Creates a Line object from a DataFrame row.

        Args:
            - df_row (pd.Series): A Series representing a single row of the DataFrame.

        Returns:
            - Line: An instance of the Line class populated with the data from the DataFrame row.
        """
        return Line(
            name=str(df_row["name"]),
            energy_type=str(df_row["energy_type"]),
            fr=str(df_row["bus_from"]),
            to=str(df_row["bus_to"]),
            transmission_loss=float(df_row["transmission_loss"]),
            max_capacity=float(df_row["max_capacity"]),
            transmission_fee=(
                None
                if pd.isnull(df_row["transmission_fee"])
                else str(df_row["transmission_fee"])
            ),
        )
