import pandas as pd

from pyzefir.model.network_elements.emission_fee import EmissionFee
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class EmissionFeeParser(AbstractElementParser):
    def __init__(
        self,
        emission_type_df: pd.DataFrame,
        emission_fee_df: pd.DataFrame,
    ) -> None:
        self.emission_fee_df = emission_fee_df.set_index("year_idx")
        self.emission_type_df = emission_type_df.set_index("emission_fee").squeeze()

    def create(self) -> tuple[EmissionFee, ...]:
        return tuple(
            EmissionFee(
                name=name,
                price=self.emission_fee_df[name],
                emission_type=self.emission_type_df[name],
            )
            for name in self.emission_fee_df.columns
        )
