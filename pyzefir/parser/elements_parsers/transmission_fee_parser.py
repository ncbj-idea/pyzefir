import pandas as pd

from pyzefir.model.network_elements import TransmissionFee
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class TransmissionFeeParser(AbstractElementParser):
    """
    Parses transmission fee data from a DataFrame to create TransmissionFee objects.

    This class transforms a DataFrame containing transmission fees indexed by hour into
    a tuple of TransmissionFee instances, where each instance represents the fee associated
    with a specific transmission line.
    """

    def __init__(self, transmission_fee_df: pd.DataFrame) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - transmission_fee_df (pd.DataFrame): DataFrame containing transmission fee information, indexed by hour.
        """
        self.transmission_fee_df = transmission_fee_df

    def create(self) -> tuple[TransmissionFee, ...]:
        """
        Creates TransmissionFee objects from the transmission fee DataFrame.

        This method sets the index of the transmission fee DataFrame to the hour index and
        creates a tuple of TransmissionFee instances for each column in the DataFrame.

        Returns:
            - tuple[TransmissionFee, ...]: A tuple of TransmissionFee instances created from the input DataFrame.
        """
        self.transmission_fee_df.set_index("hour_idx", inplace=True)

        return tuple(
            TransmissionFee(name=str(name), fee=self.transmission_fee_df[name])
            for name in self.transmission_fee_df.columns
        )
