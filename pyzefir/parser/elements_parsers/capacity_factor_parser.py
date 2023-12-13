import pandas as pd

from pyzefir.model.network_elements import CapacityFactor
from pyzefir.parser.elements_parsers.element_parser import AbstractElementParser


class CapacityFactorParser(AbstractElementParser):
    def __init__(
        self,
        capacity_factors_df: pd.DataFrame,
    ):
        self.capacity_factors_df = capacity_factors_df.copy(deep=True)

    def create(self) -> tuple[CapacityFactor, ...]:
        self.capacity_factors_df.set_index("hour_idx", inplace=True, drop=True)

        capacity_factors = []
        for col in self.capacity_factors_df.columns:
            capacity_factor = CapacityFactor(
                name=col, profile=self.capacity_factors_df[col]
            )
            capacity_factors.append(capacity_factor)
        return tuple(capacity_factors)
