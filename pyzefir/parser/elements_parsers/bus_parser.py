import pandas as pd

from pyzefir.model.network_elements import Bus
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class BusParser(AbstractElementParser):
    def __init__(
        self,
        bus_df: pd.DataFrame,
    ) -> None:
        self.bus_df = bus_df

    def create(self) -> tuple[Bus, ...]:
        return tuple(
            Bus(name=row["name"], energy_type=row["energy_type"])
            for row in self.bus_df.to_dict(orient="records")
        )
