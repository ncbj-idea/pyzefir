import pandas as pd

from pyzefir.model.network_elements import TransmissionFee
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class TransmissionFeeParser(AbstractElementParser):
    def __init__(self, transmission_fee_df: pd.DataFrame) -> None:
        self.transmission_fee_df = transmission_fee_df

    def create(self) -> tuple[TransmissionFee, ...]:
        self.transmission_fee_df.set_index("hour_idx", inplace=True)

        return tuple(
            TransmissionFee(name=name, fee=self.transmission_fee_df[name])
            for name in self.transmission_fee_df.columns
        )
