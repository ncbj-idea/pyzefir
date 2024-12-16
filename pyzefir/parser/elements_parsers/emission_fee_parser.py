import pandas as pd

from pyzefir.model.network_elements.emission_fee import EmissionFee
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class EmissionFeeParser(AbstractElementParser):
    """
    Parses and processes data to create instances of EmissionFee.

    This class takes DataFrames containing emission fee data and emission type data
    and processes them to generate EmissionFee instances. Each EmissionFee object is
    constructed from the input data, which includes the name, price, and associated
    emission type for each fee.
    """

    def __init__(
        self,
        emission_type_df: pd.DataFrame,
        emission_fee_df: pd.DataFrame,
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            - emission_type_df (pd.DataFrame): DataFrame mapping emission fee names to their types.
            - emission_fee_df (pd.DataFrame): DataFrame containing emission fee data indexed by year.
        """
        self.emission_fee_df = emission_fee_df.set_index("year_idx")
        self.emission_type_df = emission_type_df.set_index("emission_fee").squeeze()

    def create(self) -> tuple[EmissionFee, ...]:
        """
        Creates and returns a tuple of EmissionFee instances.

        This method processes the emission fee DataFrame to create EmissionFee objects.
        It extracts the necessary attributes such as name, price, and emission type from
        the DataFrames, ensuring that the prices are converted to float for each emission fee.

        Returns:
            - tuple[EmissionFee, ...]: A tuple containing the created EmissionFee instances.
        """
        return tuple(
            EmissionFee(
                name=str(name),
                price=self.emission_fee_df[name].astype(float),
                emission_type=str(self.emission_type_df[name]),
            )
            for name in self.emission_fee_df.columns
        )
