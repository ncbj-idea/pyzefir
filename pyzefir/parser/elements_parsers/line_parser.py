import pandas as pd

from pyzefir.model.network_elements import Line
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class LineParser(AbstractElementParser):
    def __init__(self, line_df: pd.DataFrame) -> None:
        self.line_df = line_df

    def create(self) -> tuple[Line, ...]:
        lines = self.line_df.apply(
            self._create_line,
            axis=1,
        )
        return tuple(lines)

    @staticmethod
    def _create_line(df_row: pd.Series) -> Line:
        return Line(
            name=df_row["name"],
            energy_type=df_row["energy_type"],
            fr=df_row["bus_from"],
            to=df_row["bus_to"],
            transmission_loss=df_row["transmission_loss"],
            max_capacity=df_row["max_capacity"],
            transmission_fee=None
            if pd.isnull(df_row["transmission_fee"])
            else df_row["transmission_fee"],
        )
